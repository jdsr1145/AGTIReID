import logging
from re import L
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .agtireid import AGTIReID
from .tagpedes import TAGPEDES

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid, 'AGTIReID': AGTIReID, 'TAGPEDES': TAGPEDES}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
            # try:
            #     # Check if v contains None values (if v is a list) or is None itself
            #     if v is None or (isinstance(v, list) and any(x is None for x in v)):
            #         print(f"!! ERROR DETECTED !! Key: '{k}' contains None values.")
            #         print(f"Value content: {v}")
                    
            #     batch_tensor_dict.update({k: torch.tensor(v)})
            # except RuntimeError as e:
            #     print(f"\nCRASH REPORT:")
            #     print(f"Failed to convert key '{k}' to tensor.")
            #     print(f"Value type: {type(v)}")
            #     print(f"Value content: {v}")
            #     raise e
        elif torch.is_tensor(v[0]):
            batch_tensor_dict.update({k: torch.stack(v)})
        # is list
        elif isinstance(v[0], list):
            # print(v, flush=True)
            list_of_tensors = []
            for i in range(len(v[0])):
                sublist = ([x[i] for x in v])
                list_of_tensors.append(torch.stack(sublist))
                # convert to tensor
            batch_tensor_dict.update({k: list_of_tensors})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir, dualstream = args.dualstream, source_aerial = args.source_aerial)
    num_classes = len(dataset.train_id_container)
    
    if args.training:
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

        # print("DEBUG: ", dataset.train[0], flush=Trues)
        # # HACK
        # dataset.train = dataset.train[:100]
        train_set = ImageTextDataset(dataset.train,args,
                                train_transforms,
                            text_length=args.text_length)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)

            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        elif args.sampler == 'random':
            # # TODO add distributed condition
            logger.info('using random sampler')
            # train_loader = DataLoader(train_set,
            #                           batch_size=args.batch_size,
            #                           shuffle=True,
            #                           num_workers=num_workers,
            #                           collate_fn=collate)
            if args.distributed:
                train_sampler = DistributedSampler(
                    train_set, shuffle=True, drop_last=True  # drop_last 推荐，避免最后空批
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=args.batch_size,
                    sampler=train_sampler,      # 用 sampler
                    shuffle=False,              # 分布式时必须 False
                    num_workers=num_workers,
                    collate_fn=collate,
                )
            else:
                train_loader = DataLoader(
                    train_set,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    collate_fn=collate,
                )
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)
        if args.enable_text_aerial:
            test_txt_set_aerial = TextDataset(ds['image_pids'], ds['captions_aerial'], text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        if args.enable_text_aerial:
            # print("DEBUG: test_txt_set_aerial shape", len(test_txt_set_aerial), flush=True)
            val_txt_loader_aerial = DataLoader(test_txt_set_aerial,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        return test_img_loader, test_txt_loader, val_txt_loader_aerial, num_classes