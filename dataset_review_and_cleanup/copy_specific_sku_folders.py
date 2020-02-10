from pathlib import Path
import shutil

import fire
from tqdm import tqdm


def _get_distinct_sku_folders(indices_path):
    with open(indices_path) as f:
        return {Path(l.strip()).parts[0] for l in f.readlines()}


def copy_folders(src_root_dir, indices_path, dst_root_dir):
    distinct_sku_folders = _get_distinct_sku_folders(indices_path)
    src_root_dir = Path(src_root_dir)
    dst_root_dir = Path(dst_root_dir)
    dst_root_dir.mkdir(parents=True, exist_ok=True)
    for sku_folder in tqdm(distinct_sku_folders):
        shutil.copytree(src_root_dir / sku_folder, dst_root_dir / sku_folder)


if __name__ == '__main__':
    fire.Fire(copy_folders)
