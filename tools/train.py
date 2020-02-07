from math import ceil
from pathlib import Path
import pandas as pd
from fastai.vision import *
from efficientnet_pytorch import EfficientNet


def read_train_val_indices(train_indices_path, val_indices_path):
    train_df = pd.read_csv(train_indices_path, header=None, sep=" ", names=["name", "label"])
    val_df = pd.read_csv(val_indices_path, header=None, sep=" ", names=["name", "label"])
    train_df['is_valid'] = False
    val_df['is_valid'] = True
    return pd.concat([train_df, val_df])


image_root_dir = Path("/data2/datasets/clobotics/ccth/images/cropped/versions/train20200129_val20200117_test20191122/for-image-quality-model/round3")
train_indices = Path("/data2/datasets/clobotics/ccth/indices/versions/train20200129_val20200117_test20191122/round3/trainval.txt")
val_indices = Path("/data2/datasets/clobotics/ccth/indices/versions/train20200129_val20200117_test20191122/round3/trainval.txt")
n_max_epochs = 3
batch_size = 96
max_lr = 9e-4
tfms = get_transforms(
    do_flip=True,  # default True
    flip_vert=False,  # default False
    max_rotate=15.0,  # default 10.0
    max_zoom=1.0,  # default 1.1
    max_lighting=0.2,  # default 0.2
    max_warp=0.2,  # default 0.2
    p_affine=0.75,  # default 0.75
    p_lighting=0.75,  # default 0.75
)

dataset = read_train_val_indices(train_indices, val_indices)

data_bunch = ImageList \
    .from_df(dataset, image_root_dir) \
    .split_from_df() \
    .label_from_df() \
    .transform(tfms, size=224, resize_method=ResizeMethod.PAD, padding_mode='zeros') \
    .databunch(bs=batch_size, num_workers=4) \
    .normalize(imagenet_stats)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
learn = Learner(data_bunch, model, metrics=accuracy, path="/tmp/fastai-models")
learn.fit_one_cycle(n_max_epochs, max_lr=max_lr)
learn.export()
