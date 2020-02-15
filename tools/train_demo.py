from fastai.vision import *
from fastai.vision.models.wrn import wrn_22
from fastai.distributed import *
from efficientnet_pytorch import EfficientNet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

path = untar_data(URLs.MNIST_SAMPLE)
ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=128).normalize(cifar_stats)
model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=2)
learn = Learner(data, model, metrics=accuracy).to_distributed(args.local_rank)
learn.fit_one_cycle(10, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)
