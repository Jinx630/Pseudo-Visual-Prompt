import os
from os.path import join
from re import L
import pickle5 as pickle
import random
import torch
import json
from tqdm import tqdm
from clip import clip

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from pycocotools.coco import COCO

from .data_helpers import *

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

object_categories = coco_object_categories
classname_synonyms = coco_classname_synonyms

clsname2idx_ = {}
nameset_compound = set()
nameset = set()
for idx, synset in enumerate(classname_synonyms):
    for n in synset:
        clsname2idx_[n] = idx
        if ' ' in n:
            nameset_compound.add(n)
            m = n.replace(' ', '')
            clsname2idx_[m] = idx
            nameset.add(m)
        else:
            nameset.add(n)

@DATASET_REGISTRY.register()
class COCO2014_distill(DatasetBase):
    def __init__(self, cfg):

        global object_categories

        self.dataset_dir = 'COCO'
        self.use_chatglm = cfg.DATASET.use_chatglm
        cls_num = len(object_categories)
    
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        coco_instance_json_file = os.path.join(self.dataset_dir, "annotations/instances_val2014.json")

        coco = COCO(coco_instance_json_file)
        self.valset_ids = coco.getImgIds()
        
        train = []

        instance_info = {}
        with open(coco_instance_json_file, 'r') as f:
            instance_info = json.load(f)

        clsid2clsidx = {}
        for idx, cat_info in enumerate(instance_info["categories"]):
            clsid2clsidx[cat_info['id']] = idx

        test_imgdir = [self.dataset_dir + '/val2014/{}'.format(coco.loadImgs(ids = imgid)[0]['file_name']) for imgid in self.valset_ids]
        test_label = torch.zeros((len(self.valset_ids), cls_num), dtype=torch.long)
        for idx, imgid in enumerate(self.valset_ids):
            annIds = coco.getAnnIds(imgIds = imgid)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                tmp_idx = clsid2clsidx[ann['category_id']]
                test_label[idx, tmp_idx] = 1
        test = []
        for i in tqdm(range(len(self.valset_ids))):
            item_ = Datum(impath=test_imgdir[i], label=test_label[i], classname='')
            test.append(item_)

        
        if not cfg.EVAL_ONLY:
            if self.use_chatglm:
                with open(cfg.DATASET.TRAIN_DATA, "r", encoding='utf-8') as f:
                    generate_data = []
                    for ann in f.readlines():
                        generate_data.append(ann)
                    # generate_data = generate_data[::10]
                    print("captions_generate nums:", len(generate_data))
            else:
                caption_feat_root = os.getcwd()
                with open(join(caption_feat_root, 'coco_caption_text_embed_sampled_idx.pkl'), 'rb') as f:
                    sample_capid = pickle.load(f)
                coco_root = self.dataset_dir
                coco_caption_json_file = os.path.join(coco_root, "annotations/captions_train2017.json")
                caption_info = {}
                with open(coco_caption_json_file, 'r') as f:
                    caption_info = json.load(f)

                anno_id2path = {}
                for i in caption_info["annotations"]:
                    anno_id2path[i["id"]] = i
                # print(i.keys())
                print("captions_train2017 nums:", len(anno_id2path))
                generate_data = []
                for i, capid in enumerate(tqdm(sample_capid)):
                    generate_data.append(anno_id2path[capid]['caption'].lower())

            # generate_data = generate_data[:1000]
            def get_wordnet_pos(tag):
                if tag.startswith('J'):
                    return wordnet.ADJ
                elif tag.startswith('V'):
                    return wordnet.VERB
                elif tag.startswith('N'):
                    return wordnet.NOUN
                elif tag.startswith('R'):
                    return wordnet.ADV
                else:
                    return None

            word_based_caption = []
            visit = []
            capid_empty_filter = set()
            wnl = WordNetLemmatizer()
            for i, mycap in enumerate(tqdm(generate_data)):
                cap = mycap.lower()
                noum_list = word_tokenize(cap)
                tagged_sent = pos_tag(noum_list) 

                lemmas_sent = []
                for tag in tagged_sent:
                    wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                    lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

                cap = ' ' + ' '.join(lemmas_sent) + ' '

                L = [0] * cls_num
                flag = 0
                for name in nameset_compound:
                    name_ = ' ' + name + ' '
                    if (name_ in cap):
                        L[clsname2idx_[name]] = 1
                        flag = 1
                        cap = cap.replace(name_, ' ')
                for name in nameset:
                    name_ = ' ' + name + ' '
                    if (name_ in cap):
                        L[clsname2idx_[name]] = 1
                        flag = 1
                        cap = cap.replace(name_, ' ')

                word_based_caption.append(L)
                if flag: #True
                    visit.append(0)
                else:
                    visit.append(1)
            print('===== Filtered by words all captions, num of captions contains object {}, num of caption empty {} ====='.format(len(word_based_caption), len(capid_empty_filter)))

            prompts = torch.cat([clip.tokenize(p) for p in generate_data])
            train = []
            for i, mycap in enumerate(generate_data):
                if visit[i]:
                    continue
                item_ = (prompts[i], torch.tensor(word_based_caption[i]))
                train.append(item_)
            print("===== Caption Distill Data: {} nums of word filtered caption  =====".format(len(word_based_caption)))
        # Add -------------------------------------------------------------------------------
        # default template
        default_prompt_num = 200
        for i in range(cls_num):
            label = [0] * cls_num
            label[i] = 1
            tmp_p = clip.tokenize(prompt_template.format(object_categories[i]))[0]
            for j_ in range(default_prompt_num - 1):
                train.append((tmp_p, torch.tensor(label)))
            
            for cur_temp in IMAGENET_TEMPLATES:
                tmp_p = clip.tokenize(cur_temp.format(object_categories[i]))[0]
                train.append((tmp_p, torch.tensor(label)))

        else:
            super().__init__(train_x=train, val=test[1::10], test=test, \
                num_classes=len(object_categories), classnames=object_categories, \
                lab2cname={idx: classname for idx, classname in enumerate(object_categories)})
