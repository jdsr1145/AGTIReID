'''
test Homography STN
'''

import os
import os.path as op
import torch
import numpy as np
import random
import time


from datasets import build_dataloader
from processor.processor import do_train, do_inference
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model, HomographySTN
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

import warnings
warnings.filterwarnings("ignore")

# TODO: Fix seed

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # TODO: rerun Re-ranking and VDT, run View-loss
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}_{args.loss_names}')
    logger = setup_logger('RDE', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)
    # if not os.path.isdir(args.output_dir+'/img'):
    #     os.makedirs(args.output_dir+'/img')
    os.makedirs(args.output_dir+'/img', exist_ok=True)
    # get image-text pair datasets dataloader

    # if 'ICFG-PEDES' not in args.dataset_name: #fixed
    #     args.val_dataset = 'val'
        
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    stn = HomographySTN(in_channels=3, out_h=256, out_w=128)
    stn = stn.to(device)
    from PIL import Image
    img = Image.open('/data2/qianruiheng/MyFolder_qrh/AGReID/AGTIReID/data/imgs/test/D1_20_000345_0_train.png').convert('RGB')
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),              # gives [0,1]
    ])

    x = transform(img).unsqueeze(0).cuda()   # (1,3,H,W)
    warped, info = stn(x)
    # warped: (B,3,H,W) in [0,1]
    warped_img = warped[0].detach().cpu()

    # convert tensor → PIL
    to_pil = T.ToPILImage()
    pil_img = to_pil(warped_img)

    pil_img.save("data/output/warped_output.jpg")

