from pathlib import Path
import shutil

from tqdm import tqdm
import fire
import pandas as pd
import numpy as np


def _get_prod_id_subbrand_id_mapping(meta):
    return meta.set_index("ProductId")["SubBrandId"].to_dict()


def _get_subbrand_id_subbrand_name_mapping(meta):
    return meta.set_index("SubBrandId")["SubBrand"].to_dict()


def _get_prod_id(sku_dir: Path):
    return int(sku_dir.stem.split("#")[0])


def _read_meta(meta_path):
    meta = pd.read_csv(meta_path)
    meta = meta[~pd.isnull(meta.SubBrandId)]
    meta.loc[:, "SubBrandId"] = meta.SubBrandId.astype(np.int64)
    return meta


def rearrange(sku_level_clusters_store_dir, dest_save_dir, meta_path, output_subbrand_dir_with_name=True):
    sku_level_clusters_store_dir = Path(sku_level_clusters_store_dir)
    dest_save_dir = Path(dest_save_dir)
    meta = _read_meta(meta_path)
    pid2sid = _get_prod_id_subbrand_id_mapping(meta)
    sid2sname = _get_subbrand_id_subbrand_name_mapping(meta)
    for sku_dir in tqdm(sku_level_clusters_store_dir.glob("*")):
        prod_id = _get_prod_id(sku_dir)

        try:
            sid = pid2sid[prod_id]
        except KeyError:
            print(f"No meta info found for {prod_id}, copied whole folder directly to {dest_save_dir}. ")
            shutil.copytree(sku_dir, dest_save_dir / sku_dir.stem)
            continue

        sname = f"#{sid2sname[sid]}" if output_subbrand_dir_with_name else ""
        _subbrand_dir = dest_save_dir / f"{sid}{sname}" / sku_dir.stem
        _subbrand_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sku_dir, _subbrand_dir)


if __name__ == '__main__':
    fire.Fire(rearrange)
