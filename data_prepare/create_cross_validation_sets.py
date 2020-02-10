from pathlib import Path

import fire
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def write_indices(save_path, paths, labels, img_root_dir=None):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(
            "\n".join(
                [f"{p if img_root_dir is None else p.relative_to(img_root_dir)} {l}" for p, l in zip(paths, labels)]
            )
        )


def create_cross_validation_sets(image_root_dir, cv_save_dir, image_suffix=".jpg", output_relative_path=True):
    image_root_dir = Path(image_root_dir)
    cv_save_dir = Path(cv_save_dir)
    im_paths = np.array(sorted(image_root_dir.rglob(f"*.{image_suffix.strip('.')}")))
    labels = np.array([p.relative_to(image_root_dir).parts[0] for p in im_paths])
    skf = StratifiedKFold(n_splits=5)
    for i, (train_index, val_index) in enumerate(skf.split(im_paths, labels)):
        print(f"\n------ Fold {i} ------")
        print(f"Train set distribution\n{pd.Series(labels[train_index]).value_counts()}\n")
        print(f"Validation set distribution\n{pd.Series(labels[val_index]).value_counts()}\n")
        write_indices(
            cv_save_dir / str(i) / "train.txt",
            im_paths[train_index],
            labels[train_index],
            img_root_dir=image_root_dir if output_relative_path else None,
        )
        write_indices(
            cv_save_dir / str(i) / "val.txt",
            im_paths[val_index],
            labels[val_index],
            img_root_dir=image_root_dir if output_relative_path else None,
        )


if __name__ == "__main__":
    fire.Fire(create_cross_validation_sets)
