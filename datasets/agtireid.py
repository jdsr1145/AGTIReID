import os
import json
import logging
import torch
import numpy as np
logger = logging.getLogger('RDE')

class AGTIReID():
    """
    MyOwnDataset
    
    Dataset statistics:
    - identities: ...
    - images: ...
    - captions: ...
    """ 
    def __init__(self, root='data', verbose=True, gonly=False, dualstream=False, source_aerial=""):
        super(AGTIReID, self).__init__()
        self.root = os.path.join(root, 'AGVPReID')
        self.source_aerial = source_aerial

        self.train_path = os.path.join(self.root, "train_paired.json")
        self.train_path_aerial = []
        self.train_path_aerial.append(os.path.join(self.root, self.source_aerial))
        logger.info(f"Using aerial source: {self.source_aerial}")
        self.train_path_aerial.append(os.path.join(self.root, "train_paired_ground_only_annotated_template.json"))
        logger.info(f"Using paired aerial annotations: train_paired_ground_only_annotated_template.json")
        self.test_path_ground = os.path.join(self.root, "test_ground.json")
        self.test_path_aerial = os.path.join(self.root, "test_aerial_ver2.json")
        self.dualstream = dualstream

        self.train, self.val, self.test, self.train_id_container = self._process_dir()

    def _process_dir(self):
        logger = logging.getLogger('RDE')
        if not self.dualstream:
            '''Train loading'''
            with open(self.train_path, 'r') as f:
                annotations = json.load(f)
            if 'DEBUG' in os.environ:
                annotations = annotations[:100]
            logger.info(f"Train path: {self.train_path}")
            all_pids = set(item[0]['id'] for item in annotations)
            pid_to_idx = {pid: i for i, pid in enumerate(sorted(list(all_pids)))}
            logger.info(f"Found {len(all_pids)} unique person IDs in the training set.")
            
            train_data = []
            train_pids = set()
            views = []
            ppo = False
            # TODO: fix image_idx
            for image_idx, item in enumerate(annotations):
                item_a = item[1]
                item = item[0]
                pid = item['id']
                file_path = item['file_path']
                file_path = file_path.split('AGVPReID/')[1]
                file_path = os.path.join(self.root, file_path)
                if ppo==False:
                    ppo=True
                    logger.info(f"An example image path: {file_path}")
                class_idx = pid_to_idx[pid]
                view = 'aerial' if 'C4' in file_path or "C5" in file_path else 'ground'
                for caption in item['captions']:
                    train_data.append([class_idx, image_idx*2, file_path, caption])
                    train_pids.add(class_idx)
                    views.append(view)
                file_path_a = item_a['file_path']
                file_path_a = file_path_a.split('AGVPReID/')[1]
                file_path_a = os.path.join(self.root, file_path_a)
                train_data.append([class_idx, image_idx*2+1, file_path_a, caption])
            logger.info(f"Loaded {len(train_data)} total training instances (image-caption pairs).")

        # dict_idx = {}
        # for i, item in enumerate(train_data):
        #     class_idx, image_idx, file_path, caption = item
        #     view = views[i]
        #     dict_idx.setdefault(class_idx, {'aerial': [], 'ground': []})
        #     dict_idx[class_idx][view].append((image_idx, file_path, caption))
        
        # # '''Loading logic 1'''
        # # # TODO: train_pids 在丢弃情况下仍保持连续
        # # train_data_grouped = []
        # # train_pids_grouped = set()
        # # ids_with_both_views = 0
        # # for item in dict_idx:
        # #     aerials = dict_idx[item]['aerial']
        # #     grounds = dict_idx[item]['ground']
        # #     len_aerials = len(aerials)
        # #     len_grounds = len(grounds)
        # #     if len_aerials == 0 or len_grounds == 0:
        # #         continue
        # #     ids_with_both_views += 1
        # #     for i in range(max(len_aerials, len_grounds)):
        # #         aerial = aerials[i % len_aerials]
        # #         ground = grounds[i % len_grounds]
        # #         train_data_grouped.append([item, aerial[0], aerial[1], ground[0], ground[1], aerial[2]])
        # #         train_pids_grouped.add(item)

        # # valid_pids = sorted(list(train_pids_grouped))
        # # pid_remap = {old_pid: new_pid for new_pid, old_pid in enumerate(valid_pids)}
        # # remapped_train_data = []
        # # for sample in train_data_grouped:
        # #     old_pid = sample[0]
        # #     new_pid = pid_remap[old_pid]
        # #     remapped_train_data.append([new_pid] + sample[1:])
        
        # # train_data_grouped = remapped_train_data
        # # train_pids_grouped = set(pid_remap.values())

        # # logger.info(f"Found {ids_with_both_views} IDs with both aerial and ground views.")
        # # logger.info(f"Created {len(train_data_grouped)} paired training samples.")

        # '''Loading logic 2'''
        # logger.info(f"AGTIReID: using unpaired training samples.")
        # train_data_grouped = train_data
        # train_pids_grouped = train_pids
        if self.dualstream:
            logger = logging.getLogger('RDE')
            annotations_list = []
            for item in self.train_path_aerial:
                with open(item, 'r') as f:
                    annotations_aerial = json.load(f)
                aerial = []
                for item in annotations_aerial:
                    if "filtered" in item:
                        aerial.append(item['processed_text'])
                    else:
                        aerial.append(item['captions'][0])
                # print("DEBUG: ", len(aerial))
                annotations_list.append(aerial)
            with open(self.train_path, 'r') as f:
                annotations = json.load(f)
            all_pids = set(item[0]['id'] for item in annotations)
            pid_to_idx = {pid: i for i, pid in enumerate(sorted(list(all_pids)))}
            logger.info(f"Found {len(all_pids)} unique person IDs in the training set.")
            train_data = []
            train_pids = set()
            views = []
            for image_idx, item in enumerate(annotations):
                item_a = item[1]
                item = item[0]
                pid = item['id']
                file_path = item['file_path']
                file_path = file_path.split('AGVPReID/')[1]
                file_path = os.path.join(self.root, file_path)
                class_idx = pid_to_idx[pid]
                caption = item['captions'][0]  # Use only the first caption from ground view
                
                pid_a = item_a['id']
                file_path_a = item_a['file_path']
                file_path_a = file_path_a.split('AGVPReID/')[1]
                file_path_a = os.path.join(self.root, file_path_a)
                caption_a = []
                for annotations in annotations_list:
                    caption_a.append(annotations[image_idx])  # Use only the first caption from aerial view
                train_data.append([class_idx, image_idx, file_path, caption, file_path_a, caption_a])
                train_pids.add(class_idx)
            logger.info(f"Loaded {len(train_data)} total training instances (image-caption pairs).")

        '''Test loading'''
        image_pids = []
        image_paths = []
        caption_pids = []
        captions = []
        captions_aerial = []
    
        with open(self.test_path_aerial, 'r') as f:
            data_image = json.load(f)
            logger.info(f"Gallery path: {self.test_path_aerial}")
        for item in data_image:
            image_pids.append(item['id'])
            image_paths.append(os.path.join(self.root, item['file_path'].split('AGVPReID/')[1]))

        with open(self.test_path_ground, 'r') as f:
            data_text = json.load(f)
            logger.info(f"Query path: {self.test_path_ground}")
        for item in data_text:
            person_id = item['id']
            for caption in item['captions']:
                caption_pids.append(person_id)
                captions.append(caption)   

        with open(self.test_path_aerial, 'r') as f:
            data_text_aerial = json.load(f)
            logger.info(f"Query Aerial path: {self.test_path_aerial}")
        for item in data_text_aerial:
            captions_aerial.append(item['captions'][0])
        
        # The code expects test/val data to be in a dictionary format
        val = test = {
            'image_pids': image_pids,
            'img_paths': image_paths,
            'caption_pids': caption_pids,
            'captions': captions,
            'captions_aerial': captions_aerial,
        }
        return train_data, val, test, train_pids