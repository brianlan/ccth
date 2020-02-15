from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm
from fastai.vision import *
from efficientnet_pytorch import EfficientNet


batch_size = 8
model_path = "/datadrive/rlan/models/bestmodel_2"
# train_class_path = "/datadrive/rlan/datasets/clobotics/ccth/indices/versions/train20200129_val20200117_test20191122/subbrand-classification/train_class.txt"
test_image_root_dir = "/datadrive/rlan/datasets/clobotics/ccth/images/cropped/versions/train20200129_val20200117_test20191122"
test_label_path = "/tmp/test.txt"
test_data = pd.read_csv(test_label_path, header=None, sep=" ", names=["name", "label"])
test_data.loc[:, "name"] = test_data.name.apply(lambda x: Path(x).relative_to(test_image_root_dir))

# with open(train_class_path) as f:
#     train_classes = [l.strip() for l in f]

# def train_classes()


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

test_image_list = ImageList.from_df(test_data, test_image_root_dir)
# data_bunch = (
#     test_image_list
#     .split_none()
#     .label_from_df(label_cls=train_classes)
#     .transform(tfms, size=256, resize_method=ResizeMethod.PAD, padding_mode="zeros")
#     .databunch(bs=batch_size, num_workers=4)
#     .normalize(imagenet_stats)
# )

data_bunch = (
    test_image_list
    .split_none()
    .label_from_df()
    .transform(tfms, size=256, resize_method=ResizeMethod.PAD, padding_mode="zeros")
    .databunch(bs=batch_size, num_workers=4)
    .normalize(imagenet_stats)
)

model = EfficientNet.from_name("efficientnet-b5", override_params={"num_classes": 197})
learn = Learner(data_bunch, model, metrics=accuracy)
learn = learn.load(model_path)
learn.data.add_test(test_image_list)
fim = open_image('/home/rlan/datasets/clobotics/ccth/images/cropped/versions/train20200129_val20200117_test20191122/val/1062929/dd30d63b9a23a7ab5802563e4dda3853_6_1062929_0.58444_0.81621_0.64509_0.90217_val.jpg')

preds, y = learn.get_preds(ds_type=DatasetType.Test)

print(learn.data.test_ds.c2i)
print(preds.argmax(dim=1))
print(y)
