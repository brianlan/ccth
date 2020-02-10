from pathlib import Path

import fire
import pandas as pd
import numpy as np


def _read_meta(meta_path):
    meta = pd.read_csv(meta_path).fillna(-1)
    meta.loc[:, "SubBrandId"] = meta.SubBrandId.fillna(-1).astype(np.int64)
    return meta


def _get_prod_id(im_path: Path):
    return int(im_path.stem.split("_")[-1])


def create_pos_class_indices_and_labels(im_paths, to_exclude):
    pos_sku_paths_and_labels = [[p, _get_prod_id(p)] for p in im_paths if _get_prod_id(p) not in to_exclude]
    print(f"Finished construction of positive classes.")
    return pos_sku_paths_and_labels


def create_neg_class_indices_and_labels(im_paths, neg_pool, n_neg_samples):
    neg_sku_paths = [p for p in im_paths if _get_prod_id(p) in neg_pool]
    selected_neg = np.random.choice(neg_sku_paths, min(len(neg_sku_paths), n_neg_samples), replace=False)
    neg_sku_paths_and_labels = [[p, 1] for p in selected_neg]
    print(f"Finished construction of negative classes.")
    return neg_sku_paths_and_labels


def _write_indices_with_label(paths_and_labels, save_path):
    pd.DataFrame(paths_and_labels, columns=["p", "l"]).to_csv(save_path, sep=" ", header=False, index=False)
    print(f"Finished writing indices file to {save_path}")


def _write_relative_path_indices(paths_and_labels, img_root_dir, save_path):
    with open(save_path, "w") as f:
        f.write("\n".join([str(Path(p[0]).relative_to(img_root_dir)) for p in paths_and_labels]))
    print(f"Finished writing indices file to {save_path}")


def write_indices(pos_paths_and_labels, neg_paths_and_labels, img_root_dir, save_dir, dataset_type):
    print(f"Writing indices (with label)..")
    _write_indices_with_label(pos_paths_and_labels, save_dir / f"{dataset_type}_pos.txt")
    _write_indices_with_label(neg_paths_and_labels, save_dir / f"{dataset_type}_neg.txt")
    _write_indices_with_label(pos_paths_and_labels + neg_paths_and_labels, save_dir / f"{dataset_type}.txt")

    print(f"Writing indices (relative path, without label)..")
    _write_relative_path_indices(pos_paths_and_labels, img_root_dir, save_dir / f"{dataset_type}_pos_indices.txt")
    _write_relative_path_indices(neg_paths_and_labels, img_root_dir, save_dir / f"{dataset_type}_neg_indices.txt")
    _write_relative_path_indices(
        pos_paths_and_labels + neg_paths_and_labels, img_root_dir, save_dir / f"{dataset_type}_indices.txt"
    )


def create(img_root_dir, meta_path, result_indices_save_dir, dataset_type, img_suffix=".jpg", n_neg_to_sample=100000):
    assert dataset_type in ["train", "val", "test"], f"Valid dataset_type are train|val|test, but {dataset_type} give."
    img_root_dir = Path(img_root_dir)
    result_indices_save_dir = Path(result_indices_save_dir)
    meta = _read_meta(meta_path)
    category_ids = meta[meta.ProductId == meta.CategoryId].ProductId.tolist()
    subbrand_other_ids = meta[meta.ProductId == meta.SubBrandId].ProductId.tolist()
    im_paths = list(img_root_dir.rglob(f"*.{img_suffix.strip('.')}"))
    print(f"Finished reading image paths.")
    pos_paths_and_labels = create_pos_class_indices_and_labels(im_paths, category_ids + subbrand_other_ids + [1])
    neg_paths_and_labels = create_neg_class_indices_and_labels(
        im_paths, subbrand_other_ids + [1], n_neg_to_sample if dataset_type == "train" else len(im_paths)
    )
    write_indices(pos_paths_and_labels, neg_paths_and_labels, img_root_dir, result_indices_save_dir, dataset_type)


if __name__ == "__main__":
    fire.Fire(create)
