from math import ceil
from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm
from fastai.vision import *
from efficientnet_pytorch import EfficientNet


# learn = load_learner(mnist, test=ImageList.from_folder(mnist/'test'))

# tfms = get_transforms(do_flip=False)
# data_bunch = ImageList.from_folder("/tmp/val-lite").split_none().label_from_folder().transform([], size=224, resize_method=ResizeMethod.PAD, padding_mode='zeros').databunch().normalize(imagenet_stats)

test_img_dir = Path("/data2/datasets/clobotics/ccth/images/cropped/versions/train20200129_val20200117_test20191122/batch4")
results_save_dir = Path("/tmp/vis/batch4")
(results_save_dir / "bad").mkdir(parents=True, exist_ok=True)
(results_save_dir / "good").mkdir(parents=True, exist_ok=True)
learn = load_learner("/tmp/fastai-models")
test_data = ImageList.from_folder(test_img_dir)
learn.data.add_test(test_data)

preds, y = learn.get_preds(ds_type=DatasetType.Test)

class_0_index = (preds.argmax(dim=1) == 0).nonzero().flatten()
class_1_index = (preds.argmax(dim=1) == 1).nonzero().flatten()

for i in tqdm(class_0_index, desc="Copy Bad"):
    _src_path = test_img_dir / test_data._relative_item_path(i)
    _dst_path = results_save_dir / "bad" / test_data._relative_item_path(i).name
    shutil.copy(_src_path, _dst_path)

for i in tqdm(class_1_index, desc="Copy Good"):
    _src_path = test_img_dir / test_data._relative_item_path(i)
    _dst_path = results_save_dir / "good" / test_data._relative_item_path(i).name
    shutil.copy(_src_path, _dst_path)
