from pathlib import Path
from collections import defaultdict
import shutil

import fire
from tqdm import tqdm
import numpy as np
import torch


def dbscan_gpu(feature, max_neighbor_dist):
    """
    :param feature: of shape num_samples x num_features
    :param max_neighbor_dist: The maximum distance between two samples for one to be considered as in the neighborhood of the other
    :return:
    """
    from cupy import asnumpy
    from cuml.cluster import DBSCAN
    db = DBSCAN(eps=max_neighbor_dist, min_samples=1).fit(feature)
    return asnumpy(db.labels_.values)


def _group_embedding_by_sku(embedding):
    groups = defaultdict(lambda: defaultdict(list))
    for i, f in zip(embedding["indices"], embedding["features"]):
        sku_name = str(Path(i).parent)
        groups[sku_name]["indices"].append(i)
        groups[sku_name]["features"].append(f)
    for sku_name in groups:
        groups[sku_name]["features"] = np.stack(groups[sku_name]["features"])
    return groups


def _read_embedding(path):
    embedding = torch.load(path)
    return {"indices": embedding["indices"], "features": embedding["features"].numpy()}


def euc_dist(x, y):
    return np.sqrt(np.square(x - y).sum())


def clustering(image_root_dir, embedding_path, clustered_save_dir, max_neighbor_dist=7.0):
    image_root_dir = Path(image_root_dir)
    clustered_save_dir = Path(clustered_save_dir)
    embedding = _read_embedding(embedding_path)
    groups = _group_embedding_by_sku(embedding)
    for sku, ebd in tqdm(groups.items()):
        indices = np.array(groups[sku]["indices"])
        features = groups[sku]["features"]

        labels = dbscan_gpu(features, max_neighbor_dist=max_neighbor_dist)
        print(f"{len(indices)} -> {labels.max()}")

        cluster_members = defaultdict(list)
        for i, l in enumerate(labels):
            cluster_members[l].append(i)

        for i, (clust_id, members) in enumerate(sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)):

            if len(members) < 2:
                ind = indices[members][0]
                fn = ind.split("/")[-1]
                _save_path = clustered_save_dir / sku / "other-clusters-merged" / fn
                _save_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(image_root_dir / ind, _save_path)
            else:
                for ind in indices[members]:
                    fn = ind.split("/")[-1]
                    _save_path = clustered_save_dir / sku / str(i) / fn
                    _save_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(image_root_dir / ind, _save_path)


if __name__ == '__main__':
    fire.Fire(clustering)
