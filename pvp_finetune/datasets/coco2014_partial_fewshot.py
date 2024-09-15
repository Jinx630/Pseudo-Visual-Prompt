import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
from pycocotools.coco import COCO
from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
import numpy as np

from .data_helpers import *

object_categories = coco_object_categories
classname_synonyms = coco_classname_synonyms

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class COCO2014_partial(DatasetBase):

    dataset_dir = "COCO"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        
        self.image_dir = self.dataset_dir
        self.split_path = os.path.join(self.dataset_dir, "split_train.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        num_shots = 8
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train_few, val_few = data["train"], data["val"]
            else:
                if os.path.exists(self.split_path):
                    train_few, val_few, test = OxfordPets.read_split(self.split_path, self.image_dir)
                train_few = self.generate_fewshot_dataset(train_few, num_shots=num_shots)
                val_few = self.generate_fewshot_dataset(val_few, num_shots=min(num_shots, 4))
                data = {"train": train_few, "val": val_few}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        ######################################################

        self.use_id = set()
        for t in train_few:
            iid = t.impath.rstrip('.jpg').split('_')[-1]
            self.use_id.add(int(iid))

        coco2014_train = os.path.join(self.dataset_dir, "annotations/instances_train2014.json")
        self.coco_train = COCO(coco2014_train)
        self.ids_train = self.coco_train.getImgIds()
        
        ## ==============================================================================
        categories = self.coco_train.loadCats(self.coco_train.getCatIds())
        categories.sort(key=lambda x: x['id'])
        num_cls = 80

        classes = {}
        coco_labels = {}
        coco_labels_inverse = {}
        for c in categories:
            coco_labels[len(classes)] = c['id']
            coco_labels_inverse[c['id']] = len(classes)
            classes[c['name']] = len(classes)

        def load_annotations(coco_, img_idlist, image_index, filter_tiny=True):
            # get ground truth annotations
            tmp_id = image_index if (img_idlist is None) else img_idlist[image_index]
            annotations_ids = coco_.getAnnIds(imgIds=tmp_id, iscrowd=False)
            annotations = []

            # some images appear to miss annotations (like image with id 257034)
            if len(annotations_ids) == 0:
                return annotations

            # parse annotations
            coco_annotations = coco_.loadAnns(annotations_ids)
            for idx, a in enumerate(coco_annotations):
                # some annotations have basically no width / height, skip them
                if filter_tiny and (a['bbox'][2] < 1 or a['bbox'][3] < 1):
                    continue
                annotations += [coco_label_to_label(a['category_id'])]

            return annotations

        def coco_label_to_label(coco_label):
            return coco_labels_inverse[coco_label]

        def label_to_coco_label(label):
            return coco_labels[label]

        def labels_list_to_1hot_partial(labels_list, class_num):
            labels_1hot = np.ones(class_num, dtype=np.float32) * (-1)
            labels_1hot[labels_list] = 1
            return labels_1hot

        coco2014_val = os.path.join(self.dataset_dir, "annotations/instances_val2014.json")
        self.coco_val = COCO(coco2014_val)
        self.ids_val = self.coco_val.getImgIds()
        
        ## ==============================================================================
        self.train_labels = []
        for idx, imgid in enumerate(self.ids_train):
            label_tmp = load_annotations(self.coco_train, None, imgid)
            label_tmp = labels_list_to_1hot_partial(label_tmp, num_cls)
            self.train_labels.append(label_tmp)
        self.train_labels = np.stack(self.train_labels, axis=0)
        self.train_labels[self.train_labels<1] = 0
        print('train_labels.shape =', self.train_labels.shape)

        train = []
        for idx, imgid in enumerate(self.ids_train):
            img_dir = self.dataset_dir + '/train2014/{}'.format(self.coco_train.loadImgs(imgid)[0]['file_name'])
            iid = img_dir.rstrip('.jpg').split('_')[-1]
            item_ = Datum(impath=img_dir, label=self.train_labels[idx], classname='')
            if int(iid) in self.use_id:
                train.append(item_)

        ## ==============================================================================
        test = []
        for idx, imgid in enumerate(self.ids_val):
            img_dir = self.dataset_dir + '/val2014/{}'.format(self.coco_val.loadImgs(imgid)[0]['file_name'])
            labels_ = labels_list_to_1hot_partial(load_annotations(self.coco_val, None, imgid, filter_tiny=False), num_cls)
            item_ = Datum(impath=img_dir, label=labels_, classname='')
            test.append(item_)

        super().__init__(train_x=train, val=test[0::20], test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})
