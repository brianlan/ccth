import pathlib

from fastai.vision import *
from efficientnet_pytorch import EfficientNet


train_indices_path = "/data2/datasets/clobotics/ccth/indices/versions/train20200129_val20200117_test20191122/subbrand-classification/train.txt"
val_indices_path = "/data2/datasets/clobotics/ccth/indices/versions/train20200129_val20200117_test20191122/subbrand-classification/val.txt"
image_root_dir = "/data2/datasets/clobotics/ccth/images/cropped/versions/train20200129_val20200117_test20191122"
batch_size = 22
model_checkpoint_dir = "/tmp/fastai-models"
pretrain_model = "/home/rlan/deploy/ccth/checkpoints/subbrand-20200212/models/bestmodel_5"
pathlib.Path(model_checkpoint_dir).mkdir(parents=True, exist_ok=True)


def read_train_val_indices(train_indices_path, val_indices_path, image_root_dir=None):
    train_df = pd.read_csv(train_indices_path, header=None, sep=" ", names=["name", "label"])
    val_df = pd.read_csv(val_indices_path, header=None, sep=" ", names=["name", "label"])
    train_df["is_valid"] = False
    val_df["is_valid"] = True
    trainval_df = pd.concat([train_df, val_df])
    if image_root_dir is not None:
        trainval_df.loc[:, "name"] = trainval_df.name.apply(lambda x: Path(x).relative_to(image_root_dir))
    return trainval_df


tfms = get_transforms(
    do_flip=True,  # default True
    flip_vert=False,  # default False
    max_rotate=10.0,  # default 10.0
    max_zoom=1.1,  # default 1.1
    max_lighting=0.2,  # default 0.2
    max_warp=0.2,  # default 0.2
    p_affine=0.75,  # default 0.75
    p_lighting=0.75,  # default 0.75
)

dataset = read_train_val_indices(train_indices_path, val_indices_path, image_root_dir=image_root_dir)
data_bunch = (
    ImageList.from_df(dataset, image_root_dir)
    .split_from_df()
    .label_from_df()
    .transform(tfms, size=256, resize_method=ResizeMethod.PAD, padding_mode="zeros")
    .databunch(bs=batch_size, num_workers=4)
    .normalize(imagenet_stats)
)

model = EfficientNet.from_name("efficientnet-b5", override_params={"num_classes": 197})
learn = Learner(data_bunch, model, metrics=accuracy, path=model_checkpoint_dir)
learn = learn.load(pretrain_model)
learn.export()
