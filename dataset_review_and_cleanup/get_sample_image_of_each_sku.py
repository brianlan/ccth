from pathlib import Path
import shutil

import pandas as pd


val = pd.read_csv("/data2/datasets/clobotics/ccth/labels/val/20200117/test_data_ccth_20200117.csv")
cnt = val["ProductId"].value_counts()
order = cnt.index.tolist()

save_dir = Path("/tmp/vis-val-sku")
save_dir.mkdir(parents=True, exist_ok=True)
for sku_dir in Path("/tmp/clust-exp").glob("*"):
    sku_id = int(str(sku_dir).split("/")[-1].split("#")[0])
    try:
        sku_img_path = list((sku_dir / "0").glob("*.jpg"))[0]
    except IndexError:
        print(f"no img found in {sku_dir}")
        continue
    rank = order.index(sku_id)
    shutil.copy(sku_img_path, save_dir / f"{rank:>04}_{sku_id}.jpg")
