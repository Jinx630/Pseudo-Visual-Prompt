from calendar import c
import os
from os.path import join
from re import L
from numpy import dtype
import pickle5 as pickle
import random
from scipy.io import loadmat
from collections import defaultdict
import torch
import json
import jsonlines
from tqdm import tqdm
from clip import clip
from clip.model import convert_weights

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

from .data_helpers import *

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

object_categories = nuswide_object_categories
classname_synonyms = nuswide_classname_synonyms

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
class nuswide_distill_limit(DatasetBase):
    def __init__(self, cfg):

        global object_categories
        cls_num = len(object_categories)
        print("loading pretrain data from: ", cfg.DATASET.train_data)
    
        with open(cfg.DATASET.train_data, "r", encoding='utf-8') as f:
            generate_data = []
            for ann in f.readlines():
                ann = ann.strip('\n')
                generate_data.append(ann)
            print("captions_generate nums:", len(generate_data))

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

        word_based_caption = [] # capid 2 cls_labels
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

        print("===== Caption Distill Data: {} nums of word filtered caption  =====".format(len(train)))
        # Add -------------------------------------------------------------------------------
        # default template
        default_prompt_num = 400
        for i in range(cls_num):
            label = [0] * cls_num
            label[i] = 1
            tmp_p = clip.tokenize(prompt_template.format(object_categories[i]))[0]
            for j_ in range(default_prompt_num - 1):
                train.append((tmp_p, torch.tensor(label)))
            
            for cur_temp in IMAGENET_TEMPLATES:
                tmp_p = clip.tokenize(cur_temp.format(object_categories[i]))[0]
                train.append((tmp_p, torch.tensor(label)))

        super().__init__(train_x=train, val=None, test=None, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})

    def read_name_list(self, path, if_split=True):
        ret = []
        with open(path, 'r') as f:
            for line in f:
                if if_split:
                    tmp = line.strip().split(' ')
                    ret.append(tmp[0])
                else:
                    tmp = line.strip()
                    ret.append(tmp)
        return ret

    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)

        print("Splitting data into 50% train, 20% val, and 30% test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y - 1, classname=c)  # convert to 0-based label
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train : n_train + n_val], label, cname))
            test.extend(_collate(impaths[n_train + n_val :], label, cname))

        return train, val, test
