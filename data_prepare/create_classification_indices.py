from pathlib import Path

import fire
import pandas as pd
import numpy as np


ignore = [1000050, 1265]


def read_meta(meta_path):
    meta = pd.read_csv(meta_path).fillna(-1)
    meta.loc[:, "SubBrandId"] = meta.SubBrandId.fillna(-1).astype(np.int64)
    return meta


def get_prod_id(im_path: Path):
    return int(im_path.parts[-2])


def read_train_classes(path):
    with open(path) as f:
        return [int(l.strip()) for l in f]


def create_sku_pos_class_indices_and_labels(im_paths, to_exclude, train_class_path=None):
    pos_sku_paths_and_labels = [[p, get_prod_id(p)] for p in im_paths if get_prod_id(p) not in to_exclude]
    if train_class_path is not None:
        train_classes = read_train_classes(train_class_path)
        pos_sku_paths_and_labels = [[p, c] for p, c in pos_sku_paths_and_labels if c in train_classes]
    print(f"Finished construction of positive classes.")
    return pos_sku_paths_and_labels


def create_sku_neg_class_indices_and_labels(im_paths, neg_pool, unknown_ratio):
    neg_sku_paths = [p for p in im_paths if get_prod_id(p) in neg_pool]
    n_neg_samples = min(len(neg_sku_paths), int(round(len(im_paths) * unknown_ratio)))

    selected_neg = np.random.choice(neg_sku_paths, min(len(neg_sku_paths), n_neg_samples), replace=False)
    neg_sku_paths_and_labels = [[p, 1] for p in selected_neg]
    print(f"Finished construction of negative classes.")
    return neg_sku_paths_and_labels


def create_subbrand_pos_class_indices_and_labels(im_paths, to_exclude, meta, train_class_path):
    pid2sid = meta.set_index("ProductId")["SubBrandId"].to_dict()  # product_id to subbrand_id mapping
    all_prod_ids = meta.ProductId.unique().tolist()
    pos_sku_paths_and_labels = []
    for p in im_paths:
        prod_id = get_prod_id(p)
        if prod_id not in to_exclude and prod_id in all_prod_ids:
            pos_sku_paths_and_labels.append([p, pid2sid[prod_id]])
    if train_class_path is not None:
        train_classes = read_train_classes(train_class_path)
        pos_sku_paths_and_labels = [[p, c] for p, c in pos_sku_paths_and_labels if c in train_classes]
    print(f"Finished construction of positive classes.")
    return pos_sku_paths_and_labels


def create_subbrand_neg_class_indices_and_labels(im_paths, neg_pool, unknown_ratio):
    neg_sku_paths = [p for p in im_paths if get_prod_id(p) in neg_pool]
    n_neg_samples = min(len(neg_sku_paths), int(round(len(im_paths) * unknown_ratio)))
    selected_neg = np.random.choice(neg_sku_paths, n_neg_samples, replace=False)
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


def _write_distinct_class(paths_and_labels, save_path):
    with open(save_path, "w") as f:
        f.write("\n".join(np.unique([str(c) for p, c in paths_and_labels])))
    print(f"Finished writing distinct classes file to {save_path}")


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

    print(f"Writing distinct classes..")
    _write_distinct_class(pos_paths_and_labels + neg_paths_and_labels, save_dir / f"{dataset_type}_class.txt")


def create(
    img_root_dir,
    meta_path,
    result_indices_save_dir,
    dataset_type,
    train_class_path=None,
    img_suffix=".jpg",
    unknown_ratio=0.05,
    for_subbrand=False,
):
    assert dataset_type in ["train", "val", "test"], f"Valid dataset_type are train|val|test, but {dataset_type} give."
    img_root_dir = Path(img_root_dir)
    result_indices_save_dir = Path(result_indices_save_dir)
    result_indices_save_dir.mkdir(parents=True, exist_ok=True)
    meta = read_meta(meta_path)
    category_ids = meta[meta.ProductId == meta.CategoryId].ProductId.tolist()
    subbrand_other_ids = meta[meta.SKUName.str.strip().str.lower().str.endswith("-other")].ProductId.tolist()
    im_paths = list(img_root_dir.rglob(f"*.{img_suffix.strip('.')}"))
    print(f"Finished reading image paths.")

    if for_subbrand:
        pos_paths_and_labels = create_subbrand_pos_class_indices_and_labels(
            im_paths, ignore + category_ids + [1], meta, None if dataset_type == "train" else train_class_path
        )
        neg_paths_and_labels = create_subbrand_neg_class_indices_and_labels(
            im_paths, [1], unknown_ratio if dataset_type == "train" else 1.0
        )
    else:
        pos_paths_and_labels = create_sku_pos_class_indices_and_labels(
            im_paths,
            ignore + category_ids + subbrand_other_ids + [1],
            None if dataset_type == "train" else train_class_path,
        )
        neg_paths_and_labels = create_sku_neg_class_indices_and_labels(
            im_paths, subbrand_other_ids + [1], unknown_ratio if dataset_type == "train" else 1.0
        )
    write_indices(pos_paths_and_labels, neg_paths_and_labels, img_root_dir, result_indices_save_dir, dataset_type)


if __name__ == "__main__":
    fire.Fire(create)
