import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.models._utils import IntermediateLayerGetter
from tqdm import tqdm
import pickle5 as pickle
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import  load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .utils import ranking_loss
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

class ImagePromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        print(f"global size: {cfg.TRAINER.global_size}, local size: {cfg.TRAINER.local_size}.")
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
        """
        Returns current learned ctx embeddings, concated with cls word embeddings.
        """
        return self.ctx, self.ctx_double, self.temperature, self.spatial_T, self.ranking_scale

class DenseCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, return_interm_layers=False):
        super().__init__()
        self.image_prompt_learner = ImagePromptLearner(cfg, classnames, clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual

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
            # ====================== Get input image feature =============================
            image_feat = self.encode_image(image)
            b, c, h, w = image_feat.shape
            x = image_feat.reshape(b, c, h * w).permute(2, 0, 1) 
            x = F.linear(x, self.v_linear_weight, self.v_linear_bias)
            x = F.linear(x, self.c_linear_weight, self.c_linear_bias)
            image_features_global, _ = self.image_encoder.attnpool(image_feat, if_pos=False)
            image_features_local = torch.cat([x, image_features_global.unsqueeze(0)], dim = 0) 

            # ====================== Get image prompt feature ============================
            prompts_image_global, prompts_image_local, temperature_image, spatial_T_image, _ = self.image_prompt_learner()
            image_prompt_global = self.encode_image_global(prompts_image_global)
            image_prompt_local = self.encode_image_global(prompts_image_local)

            # ====================== Normalization =======================================
            image_features_global = image_features_global / image_features_global.norm(dim=-1, keepdim=True)
            image_features_local  = image_features_local  / image_features_local.norm(dim=-1, keepdim=True)
            image_prompt_global   = image_prompt_global   / image_prompt_global.norm(dim=-1, keepdim=True)
            image_prompt_local    = image_prompt_local    / image_prompt_local.norm(dim=-1, keepdim=True)

            # ====================== Temperature coefficient =============================
            logit_scale_image = temperature_image.exp()  
            logit_scale_image = logit_scale_image if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50
            tmp_scale_image   = spatial_T_image.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_text  # 5 #

            # ====================== Image prompt similarity =============================
            logits_image_global = logit_scale_image * image_features_global @ image_prompt_global.t()  
            logits_image_local_part = image_features_local @ image_prompt_local.t()    
            prob_spatial_image = torch.nn.functional.softmax(logits_image_local_part * tmp_scale_image, dim=0)
            logits_image_local = torch.sum(logit_scale_image * logits_image_local_part * prob_spatial_image, dim=0)

            return logits_image_global, logits_image_local
        else:
            # ====================== Get input caption feature ===========================
            # b, l, d = text_feat.shape
            text_feat = self.text_encoder(captions, None, if_embedding=False, if_sequence=True) 
            text_features_global = text_feat[torch.arange(text_feat.shape[0]), captions.argmax(dim=-1)]  # BD
            text_features_local = text_feat.permute(1, 0, 2)  # LBD

            # ====================== Get image prompt feature ============================
            prompts_image_global, prompts_image_local, temperature_image, spatial_T_image, _ = self.image_prompt_learner()
            image_prompt_global = self.encode_image_global(prompts_image_global)
            image_prompt_local = self.encode_image_global(prompts_image_local)

            # ====================== Normalization ======================================
            text_features_global = text_features_global / text_features_global.norm(dim=-1, keepdim=True)
            text_features_global += torch.randn_like(text_features_global) * 0.04
            text_features_global = text_features_global / text_features_global.norm(dim=-1, keepdim=True)

            text_features_local = text_features_local / text_features_local.norm(dim=-1, keepdim=True)
            text_features_local += torch.randn_like(text_features_local) * 0.04
            text_features_local = text_features_local / text_features_local.norm(dim=-1, keepdim=True)
            
            image_prompt_global = image_prompt_global / image_prompt_global.norm(dim=-1, keepdim=True)
            image_prompt_local  = image_prompt_local  / image_prompt_local.norm(dim=-1, keepdim=True)
            
            # ====================== Temperature coefficient =============================
            logit_scale_image = temperature_image.exp()  
            logit_scale_image = logit_scale_image if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50
            tmp_scale_image   = spatial_T_image.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_image  # 5 #

            # ====================== Image prompt similarity =============================
            # mask irrelavent tokens
            text_mask = (captions == 0).long() * (-10000)  # BL
            
            logits_image_global = logit_scale_image * text_features_global @ image_prompt_global.t()  
            logits_image_local_part = text_features_local @ image_prompt_local.t()    
            logits_image_local_part = logits_image_local_part.permute(2, 1, 0) + text_mask[None, :, :]
            logits_image_local_part = logits_image_local_part.permute(2, 1, 0)
            prob_spatial_image = torch.nn.functional.softmax(logits_image_local_part * tmp_scale_image, dim=0)
            logits_image_local = torch.sum(logit_scale_image * logits_image_local_part * prob_spatial_image, dim=0)
            
            return logits_image_global, logits_image_local


@TRAINER_REGISTRY.register()
class Caption_distill_double(TrainerX):
    def model_inference(self, input):
        return self.model(input, if_test=True)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        return None

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
            if "image_prompt_learner" not in name and ("text_ap" not in name) and ("image_ap" not in name):
                param.requires_grad_(False)

        print("requires_grad = True Layers:")
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                print(name)

        self.model.to(self.device)
        # NOTE: only give image_prompt_learner to the optimizer
        self.optim_image = build_optimizer(self.model.image_prompt_learner, cfg.OPTIM, 0.1)
        self.sched_image = build_lr_scheduler(self.optim_image, cfg.OPTIM, 1.0, 0.001)
        self.register_model("image_prompt_learner", self.model.image_prompt_learner, self.optim_image, self.sched_image)

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
            image_global, image_local = self.model(None, image)
            if self.cfg.TRAIN.LOSSFUNC == 'double_ranking':
                loss =  ranking_loss(image_global, label, scale_ = 1.0, margin_ = 1) + \
                        0.1 * ranking_loss(image_local, label, scale_ = 1.0, margin_ = 1)
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

    def load_model(self, directory, epoch=None):
        return 
