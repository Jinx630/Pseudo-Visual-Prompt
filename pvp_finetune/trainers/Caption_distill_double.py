import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from torchvision.models._utils import IntermediateLayerGetter
from tqdm import tqdm
import pickle5 as pickle
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import  load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .utils import ranking_loss
import numpy as np

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts, if_embedding=True, if_sequence=False):
        if not if_embedding:
            tokenized_prompts = prompts
            prompts = self.token_embedding(prompts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        
        if if_sequence:
            x = x @ self.text_projection  # NLD * Dd = NLd
            return x
        else:
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # ND * Dd = Nd
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
            return x


class TextPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        classnames += ''

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.Caption.N_CTX
        ctx_init = cfg.TRAINER.Caption.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.Caption.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            
            if cfg.TRAINER.Caption.CSC:
                print("Initializing class-specific double contexts")
                ctx_vectors_double = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_double = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_double, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial double context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_double = nn.Parameter(ctx_vectors_double)  # to be optimized
        
        temperature = torch.tensor(3.0, dtype=dtype)  #  exp(3.91) = 50
        self.temperature = nn.Parameter(temperature)
        spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)
        ranking_scale = torch.tensor(4.0, dtype=dtype)  # 20
        self.ranking_scale = nn.Parameter(ranking_scale)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        
        # class agnostic token suffix
        prompts_nocls = [prompt_prefix + "."] * len(classnames)
        tokenized_prompts_nocls = torch.cat([clip.tokenize(p) for p in prompts_nocls])
        with torch.no_grad():
            embedding_nocls = clip_model.token_embedding(tokenized_prompts_nocls).type(dtype)
        self.register_buffer("token_suffix_nocls", embedding_nocls[:, 1 + n_ctx :, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.Caption.CLASS_TOKEN_POSITION

    def forward(self, neg_prompt_wcls=True):
        """
        Returns current learned ctx embeddings, concated with cls word embeddings.
        """
        ctx = self.ctx
        ctx_double = self.ctx_double
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        if ctx_double.dim() == 2:
            ctx_double = ctx_double.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        suffix_nocls = self.token_suffix_nocls

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            if neg_prompt_wcls:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,     # (n_cls, n_ctx, dim)
                        suffix_nocls,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
        else:
            raise ValueError

        return prompts, prompts_neg, self.temperature, self.spatial_T, self.ranking_scale


class ImagePromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        mean_list = [0.48145466, 0.4578275, 0.40821073]
        std_list = [0.26862954, 0.26130258, 0.27577711]
        # random initialization
        ctx_vectors = torch.empty(3, n_cls, cfg.TRAINER.global_size, cfg.TRAINER.global_size, dtype=dtype)
        for i in range(3):
            nn.init.normal_(ctx_vectors[i], std=std_list[i], mean=mean_list[i])
        ctx_vectors = ctx_vectors.permute(1, 0, 2, 3)

        ctx_vectors_double = torch.empty(3, n_cls, cfg.TRAINER.local_size, cfg.TRAINER.local_size, dtype=dtype)
        for i in range(3):
            nn.init.normal_(ctx_vectors_double[i], std=std_list[i], mean=mean_list[i])
        ctx_vectors_double = ctx_vectors_double.permute(1, 0, 2, 3)

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_double = nn.Parameter(ctx_vectors_double)  # to be optimized
        
        temperature = torch.tensor(3.0, dtype=dtype)  #  exp(3.91) = 50
        self.temperature = nn.Parameter(temperature)
        spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)
        ranking_scale = torch.tensor(4.0, dtype=dtype)  # 20
        self.ranking_scale = nn.Parameter(ranking_scale)

        self.n_cls = n_cls

    def forward(self):
        return self.ctx, self.ctx_double, self.temperature, self.spatial_T, self.ranking_scale


class Adaptor_Learner(nn.Module):
    def __init__(self, multi=4, alpha=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024 * multi, bias=False),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024 * multi, 1024, bias=False),
        )
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * x + (1.0 - self.alpha) * self.fc(x)

class DenseCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, return_interm_layers=False):
        super().__init__()    
        self.use_tta = cfg.TEST.use_tta
        self.text_prompt_learner = TextPromptLearner(cfg, classnames, clip_model)
        self.image_prompt_learner = ImagePromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.text_ap = Adaptor_Learner(multi = 1, alpha=0.6) 
        self.image_ap = Adaptor_Learner(multi = 1, alpha=0.6)

        self.model = clip_model
        self.return_interm_layers = return_interm_layers
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.visual_encoder = IntermediateLayerGetter(self.image_encoder, return_layers)
        self.positional_embedding = self.image_encoder.attnpool.positional_embedding[1::]
        self.v_linear_weight = self.image_encoder.attnpool.v_proj.weight
        self.v_linear_bias = self.image_encoder.attnpool.v_proj.bias
        self.c_linear_weight = self.image_encoder.attnpool.c_proj.weight
        self.c_linear_bias = self.image_encoder.attnpool.c_proj.bias
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg

    
    def encode_image(self, x):
        def stem(x):
            for conv, bn in [(self.visual_encoder.conv1, self.visual_encoder.bn1), \
                (self.visual_encoder.conv2, self.visual_encoder.bn2), (self.visual_encoder.conv3, self.visual_encoder.bn3)]:
                x = self.visual_encoder.relu(bn(conv(x)))
            x = self.visual_encoder.avgpool(x)
            return x

        x = x.type(self.visual_encoder.conv1.weight.dtype)
        x = stem(x)
        x = self.visual_encoder.layer1(x)
        x = self.visual_encoder.layer2(x)
        x = self.visual_encoder.layer3(x)
        x = self.visual_encoder.layer4(x)
        return x
    
    def encode_image_global(self, x):
        x = self.encode_image(x)
        x_global, _ = self.image_encoder.attnpool(x)
        return x_global

    def forward(self, image=None, captions=None, if_test=False):
        if if_test:    
            def horizontal_flip(image):
                logits_text_global_ls = []
                logits_text_local_ls = []
                logits_image_global_ls = []
                logits_image_local_ls = []
                for tta_value in range(1):
                    image_t = image
                    if (tta_value & 1) == 1:
                        image_t = torch.flip(image_t, dims = [3])
                    # ====================== Get input image feature =============================
                    image_feat = self.encode_image(image_t)
                    b, c, h, w = image_feat.shape
                    x = image_feat.reshape(b, c, h * w).permute(2, 0, 1) 
                    x = F.linear(x, self.v_linear_weight, self.v_linear_bias)
                    x = F.linear(x, self.c_linear_weight, self.c_linear_bias)

                    image_features_global, _ = self.image_encoder.attnpool(image_feat, if_pos=False)
                    image_features_local = torch.cat([x, image_features_global.unsqueeze(0)], dim = 0) 
                    
                    # ====================== Get text prompt feature =============================
                    prompts_text_global, prompts_text_local, temperature_text, spatial_T_text, _ = self.text_prompt_learner()
                    tokenized_prompts  = self.tokenized_prompts
                    text_prompt_global = self.text_encoder(prompts_text_global, tokenized_prompts)
                    text_prompt_local  = self.text_encoder(prompts_text_local, tokenized_prompts)

                    # ====================== Get image prompt feature ============================
                    prompts_image_global, prompts_image_local, temperature_image, spatial_T_image, _ = self.image_prompt_learner()
                    image_prompt_global = self.encode_image_global(prompts_image_global)
                    image_prompt_local  = self.encode_image_global(prompts_image_local)

                    # ============================== Adapter =====================================
                    image_prompt_global    = image_prompt_global    / image_prompt_global.norm(dim=-1, keepdim=True)
                    image_prompt_local     = image_prompt_local     / image_prompt_local.norm(dim=-1, keepdim=True)
                    image_features_global  = image_features_global  / image_features_global.norm(dim=-1, keepdim=True)

                    text_prompt_global     = text_prompt_global  / text_prompt_global.norm(dim=-1, keepdim=True)
                    text_prompt_local      = text_prompt_local   / text_prompt_local.norm(dim=-1, keepdim=True)
                    
                    image_prompt_global   = self.image_ap(image_prompt_global)
                    image_prompt_local    = self.image_ap(image_prompt_local)
                    image_features_global = self.image_ap(image_features_global)

                    text_prompt_global = self.text_ap(text_prompt_global)
                    text_prompt_local  = self.text_ap(text_prompt_local)
                    # ====================== Normalization =======================================
                    image_features_global = image_features_global / image_features_global.norm(dim=-1, keepdim=True)
                    image_features_local  = image_features_local  / image_features_local.norm(dim=-1, keepdim=True)
                    text_prompt_global    = text_prompt_global    / text_prompt_global.norm(dim=-1, keepdim=True)
                    text_prompt_local     = text_prompt_local     / text_prompt_local.norm(dim=-1, keepdim=True)
                    image_prompt_global   = image_prompt_global   / image_prompt_global.norm(dim=-1, keepdim=True)
                    image_prompt_local    = image_prompt_local    / image_prompt_local.norm(dim=-1, keepdim=True)

                    # ====================== Temperature coefficient =============================
                    logit_scale_text = temperature_text.exp()
                    logit_scale_text = logit_scale_text if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50 
                    tmp_scale_text   = spatial_T_text.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_image   # 40 #
                    
                    logit_scale_image = temperature_image.exp()
                    logit_scale_image = logit_scale_image if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50
                    tmp_scale_image   = spatial_T_image.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_text  # 50 #

                    # ====================== Text prompt similarity ==============================
                    logits_text_global = logit_scale_text * image_features_global @ text_prompt_global.t()   # B * C,  cls * C, = B * cls
                    logits_text_local_part = image_features_local @ text_prompt_local.t()    #  HW * B * C,  cls * C,  HW * B * cls
                    prob_spatial_text = torch.nn.functional.softmax(logits_text_local_part * tmp_scale_text, dim=0)
                    logits_text_local = torch.sum(logit_scale_text * logits_text_local_part * prob_spatial_text, dim=0)

                    # ====================== Image prompt similarity =============================
                    logits_image_global = logit_scale_image * image_features_global @ image_prompt_global.t()  
                    logits_image_local_part = image_features_local @ image_prompt_local.t()    
                    prob_spatial_image = torch.nn.functional.softmax(logits_image_local_part * tmp_scale_image, dim=0)
                    logits_image_local = torch.sum(logit_scale_image * logits_image_local_part * prob_spatial_image, dim=0)
                    
                    logits_text_global_ls.append(logits_text_global)
                    logits_text_local_ls.append(logits_text_local)
                    logits_image_global_ls.append(logits_image_global)
                    logits_image_local_ls.append(logits_image_local)

                def list_mean(x):
                    x = torch.stack(x)
                    return torch.mean(x, dim = 0)
                logits_text_global = list_mean(logits_text_global_ls)
                logits_text_local = list_mean(logits_text_local_ls)
                logits_image_global = list_mean(logits_image_global_ls)
                logits_image_local = list_mean(logits_image_local_ls)

                return [logits_text_global, logits_text_local, logits_image_global, logits_image_local], image_features_local, text_prompt_global, text_prompt_local, image_prompt_global, image_prompt_local
            
            aug_num, bs, c, h, w = image.shape
            image = image.reshape(-1, c, h, w)
            origin_score, image_features_local, text_prompt_global, text_prompt_local, image_prompt_global, image_prompt_local = horizontal_flip(image)
            final_score = []
            for idx, x in enumerate(origin_score):
                if self.use_tta:
                    global_logits, part_logits = torch.split(x.reshape(aug_num, bs, 80), dim = 0, split_size_or_sections = [1, aug_num - 1])
                    result_1 = global_logits * part_logits 
                    result_1 = torch.max(result_1, dim = 0)[0]
                    scale = 0.5
                    result_1 = torch.max(global_logits.squeeze(0) * scale, result_1) / scale

                    result_2 = x.reshape(aug_num, bs, 80)
                    result_2 = torch.max(result_2, dim = 0)[0]

                    final_score.append(0.3 * result_1 + 0.7 * result_2)
                else:
                    global_logits = x.reshape(aug_num, bs, -1)
                    result_1 = torch.max(global_logits, dim = 0)[0]
                    final_score.append(result_1)

            return tuple(final_score) + (image_features_local, text_prompt_global, text_prompt_local, image_prompt_global, image_prompt_local)
            
        else:
            # ====================== Get input caption feature ===========================
            text_feat = self.text_encoder(captions, None, if_embedding=False, if_sequence=True)
            text_features_global = text_feat[torch.arange(text_feat.shape[0]), captions.argmax(dim=-1)]  # BD
            text_features_local = text_feat.permute(1, 0, 2)  # LBD

            # ====================== Get text prompt feature =============================
            prompts_text_global, prompts_text_local, temperature_text, spatial_T_text, _ = self.text_prompt_learner()
            tokenized_prompts  = self.tokenized_prompts
            text_prompt_global = self.text_encoder(prompts_text_global, tokenized_prompts)
            text_prompt_local  = self.text_encoder(prompts_text_local, tokenized_prompts)

            # ====================== Get image prompt feature ============================
            prompts_image_global, prompts_image_local, temperature_image, spatial_T_image, _ = self.image_prompt_learner()
            image_prompt_global = self.encode_image_global(prompts_image_global)
            image_prompt_local  = self.encode_image_global(prompts_image_local)

            # ============================== Adapter =====================================
            image_prompt_global  = image_prompt_global  / image_prompt_global.norm(dim=-1, keepdim=True)
            image_prompt_local   = image_prompt_local   / image_prompt_local.norm(dim=-1, keepdim=True)

            text_prompt_global  = text_prompt_global  / text_prompt_global.norm(dim=-1, keepdim=True)
            text_prompt_local   = text_prompt_local   / text_prompt_local.norm(dim=-1, keepdim=True)
            text_features_global  = text_features_global  / text_features_global.norm(dim=-1, keepdim=True)
            
            
            image_prompt_global = self.image_ap(image_prompt_global)
            image_prompt_local  = self.image_ap(image_prompt_local)

            text_prompt_global    = self.text_ap(text_prompt_global)
            text_prompt_local     = self.text_ap(text_prompt_local)
            text_features_global = self.text_ap(text_features_global)
            # ====================== Normalization ======================================
            text_features_global  = text_features_global / text_features_global.norm(dim=-1, keepdim=True)
            text_features_global += torch.randn_like(text_features_global) * self.cfg.TRAINER.NOSIE
            text_features_global  = text_features_global / text_features_global.norm(dim=-1, keepdim=True)
            
            text_features_local = text_features_local / text_features_local.norm(dim=-1, keepdim=True)
            text_features_local += torch.randn_like(text_features_local) * self.cfg.TRAINER.NOSIE
            text_features_local = text_features_local / text_features_local.norm(dim=-1, keepdim=True)

            text_prompt_global  = text_prompt_global  / text_prompt_global.norm(dim=-1, keepdim=True)
            text_prompt_local   = text_prompt_local   / text_prompt_local.norm(dim=-1, keepdim=True)
            image_prompt_global = image_prompt_global / image_prompt_global.norm(dim=-1, keepdim=True)
            image_prompt_local  = image_prompt_local  / image_prompt_local.norm(dim=-1, keepdim=True)
            
            # ====================== Temperature coefficient =============================
            logit_scale_text = temperature_text.exp()
            logit_scale_text = logit_scale_text if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50 
            tmp_scale_text   = spatial_T_text.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_text
            
            logit_scale_image = temperature_image.exp()  
            logit_scale_image = logit_scale_image if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50
            tmp_scale_image   = spatial_T_image.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_image  # 5 #
            # ====================== Text prompt similarity ==============================
            # mask irrelavent tokens
            text_mask = (captions == 0).long() * (-10000)  # BL

            logits_text_global = logit_scale_text * text_features_global @ text_prompt_global.t()   # B * C,  cls * C, = B * cls
            logits_text_local_part = text_features_local @ text_prompt_local.t()    #  L * B * C,  cls * C,  L * B * cls
            logits_text_local_part = logits_text_local_part.permute(2, 1, 0) + text_mask[None, :, :]
            logits_text_local_part = logits_text_local_part.permute(2, 1, 0)
            prob_spatial_text = torch.nn.functional.softmax(logits_text_local_part * tmp_scale_text, dim=0)
            logits_text_local = torch.sum(logit_scale_text * logits_text_local_part * prob_spatial_text, dim=0)

            # ====================== Image prompt similarity =============================
            logits_image_global = logit_scale_image * text_features_global @ image_prompt_global.t()  
            logits_image_local_part = text_features_local @ image_prompt_local.t()    
            logits_image_local_part = logits_image_local_part.permute(2, 1, 0) + text_mask[None, :, :]
            logits_image_local_part = logits_image_local_part.permute(2, 1, 0)
            prob_spatial_image = torch.nn.functional.softmax(logits_image_local_part * tmp_scale_image, dim=0)
            logits_image_local = torch.sum(logit_scale_image * logits_image_local_part * prob_spatial_image, dim=0)
            
            tai_logits = self.logit_scale * text_prompt_global @ image_prompt_global.t()
            iat_logits = self.logit_scale * image_prompt_global @ text_prompt_global.t()
            return logits_text_global, logits_text_local, logits_image_global, logits_image_local, tai_logits, iat_logits

@TRAINER_REGISTRY.register()
class Caption_distill_double(TrainerX):
    def model_inference(self, input):
        return self.model(input, if_test=True)

    def parse_batch_test(self, batch, use_tta):
        input_org = batch["img"].unsqueeze(0)
        if use_tta:
            input_aug_fivecrop = batch["aug_img_fivecrop"].permute(1, 0, 2, 3, 4)
            input_aug_randomcrop = batch["aug_img_randomcrop"].permute(1, 0, 2, 3, 4)
            input = torch.cat([input_org, input_aug_fivecrop, input_aug_randomcrop], dim = 0)
            input = input.to(self.device)
        else:
            input = input_org.to(self.device)

        label = batch["label"]
        label = label.to(self.device)
        return input, label

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")
        
        img_feas = []
        outputs_t = []
        outputs_g = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch,self.use_tta)
            text_global, text_local, image_global, image_local, image_features_local, text_prompt_global, text_prompt_local, image_prompt_global, image_prompt_local = self.model_inference(input)
            self.evaluator.process(text_global, image_global, label, text_local, image_local)
            
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def check_cfg(self, cfg):
        assert cfg.TRAINER.Caption.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        print('==================== Building model in Caption_distill_double ======================')
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.Caption.PREC == "fp32" or cfg.TRAINER.Caption.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = DenseCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if ("text_prompt_learner" not in name) and ("image_prompt_learner" not in name) \
                and ("text_ap" not in name) and ("image_ap" not in name):
                param.requires_grad_(False)

        print("requires_grad = True Layers:")
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                print(name)

        self.model.to(self.device)
        
        # NOTE: only give text_prompt_learner & image_prompt_learner & image_adaptor & text_adaptor to the optimizer
        self.optim_text = build_optimizer(self.model.text_prompt_learner, cfg.OPTIM, cfg.TRAINER.SETLR)
        self.sched_text = build_lr_scheduler(self.optim_text, cfg.OPTIM, cfg.TRAINER.SETLR, cfg.TRAINER.SETLR_MUL)
        self.register_model("text_prompt_learner", self.model.text_prompt_learner, self.optim_text, self.sched_text)

        self.optim_image = build_optimizer(self.model.image_prompt_learner, cfg.OPTIM, cfg.TRAINER.SETLR)
        self.sched_image = build_lr_scheduler(self.optim_image, cfg.OPTIM, cfg.TRAINER.SETLR, cfg.TRAINER.SETLR_MUL)
        self.register_model("image_prompt_learner", self.model.image_prompt_learner, self.optim_image, self.sched_image)

        self.optim_image_clip = build_optimizer(self.model.image_ap, cfg.OPTIM, 1e-3)
        self.sched_image_clip = build_lr_scheduler(self.optim_image_clip, cfg.OPTIM, 5e-3, 0.1)
        self.register_model("image_ap", self.model.image_ap, self.optim_image_clip, self.sched_image_clip)

        self.optim_text_clip = build_optimizer(self.model.text_ap, cfg.OPTIM, 1e-3)
        self.sched_text_clip = build_lr_scheduler(self.optim_text_clip, cfg.OPTIM, 5e-3, 0.1)
        self.register_model("text_ap", self.model.text_ap, self.optim_text_clip, self.sched_text_clip)

        self.scaler = GradScaler() if cfg.TRAINER.Caption.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.Caption.PREC
        if prec == "amp":
            assert False, 'Not implemented.'
        else:
            text_global, text_local, image_global, image_local, tai, iat = self.model(None, image)
            ce = nn.CrossEntropyLoss(reduction = 'sum')
            ce_label = torch.arange(tai.size(1)).to(tai.device)
            if self.cfg.TRAIN.LOSSFUNC == 'double_ranking':
                rank_w = 1.0
                loss_text_global = ranking_loss(text_global, label, scale_ = 1.0, margin_ = rank_w)
                loss_text_local = 0.01 * ranking_loss(text_local, label, scale_ = 1.0, margin_ = rank_w)
                loss_image_global = ranking_loss(image_global, label, scale_ = 1.0, margin_ = rank_w)
                loss_image_local = ranking_loss(image_local, label, scale_ = 1.0, margin_ = rank_w)
                loss_ce = (ce(tai, ce_label) + ce(iat, ce_label)) / 2
                a,b,c = [1,1,1]
                loss = c*loss_text_global + c*loss_text_local + b*loss_image_global + b*loss_image_local + a*loss_ce
            else:
                assert False, 'Not implemented.'
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, image_pretrain=True, text_pretrain=True, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = f"model-{epoch}-best.pth.tar"
        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not self.cfg.EVAL_ONLY:
                if name == 'image_prompt_learner' and (not image_pretrain): 
                    continue
                if name == 'text_prompt_learner' and (not text_pretrain): 
                    continue
                if name == 'image_ap':
                    continue
                if name == 'text_ap':
                    continue
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
    


