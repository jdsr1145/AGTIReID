import os.path as op
import os
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset

import re
import json

class TAGPEDES(BaseDataset):
    """
    TAGPEDES
    """
    def __init__(self, root='data', verbose=True, gonly=False, dualstream=False, source_aerial=""):
        super(TAGPEDES, self).__init__() # Init of BaseDataset
        self.root = op.join(root, 'TAG')

        self.train_annos, self.test_annos, self.val_annos = self._split_anno()

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)

        if not self.val_annos:
            import copy
            self.val_annos = copy.deepcopy(self.test_annos)
            self.val = copy.deepcopy(self.test)
            self.val_id_container = copy.deepcopy(self.test_id_container)
        else:
            self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> TAGPEDES Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self):
        train_annos, test_annos, val_annos = [], [], []
        train_anno_path = op.join(self.root, 'train_reid.json')
        test_anno_path = op.join(self.root, 'test_reid.json')
        # # HACK
        self.logger.info("=> TAGPEDES Using aerial-view annotations")
        train_anno_path = 'data/train_reid.json'
        # test_anno_path = 'data/test_reid.json'
        with open(train_anno_path, 'r') as f:
            train_annos = json.load(f)
        with open(test_anno_path, 'r') as f:
            test_annos = json.load(f)
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        # self.logger.info("QRH: Using sr images")
        if training:
            dataset = []
            image_id = 0 # TODO: determine function of variable
            for item in annos:
                pid = int(item['id'])
                pid_container.add(pid)
                img_path = op.join(self.root, item['file_path'])
                # img_path_sr = img_path.replace('.jpg', '_sr.png')
                # if op.exists(img_path_sr):
                    # img_path = img_path_sr
                captions = item['captions'] # caption list
                cam_id = item['cam_id']
                for caption in captions:
                    caption = pre_caption(caption)
                    dataset.append((pid, image_id, img_path, caption, cam_id))
                image_id += 1
            pids_sorted = sorted(pid_container)
            assert pids_sorted == list(range(len(pids_sorted))), \
                f"pids not contiguous from 0: {pids_sorted}"
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.root, anno['file_path'])
                img_path_sr = img_path.replace('.jpg', '_sr.png')
                if op.exists(img_path_sr):
                    img_path = img_path_sr
                img_paths.append(img_path)
                image_pids.append(pid)
                for caption in anno['captions']:
                    captions.append(pre_caption(caption))
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container

def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption