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


def _group_embedding_by_sku(embedding, indices_path):
    with open(indices_path) as f:
        paths, classes = zip(*[l.strip().split() for l in f])
    order = np.argsort(paths)  # we sort according to path because get_embedding sorts the paths.
    sorted_classes = np.array(classes)[order]
    groups = defaultdict(lambda: defaultdict(list))
    for c, i, f in zip(sorted_classes, embedding["indices"], embedding["features"]):
        groups[c]["indices"].append(i)
        groups[c]["features"].append(f)
    for c in groups:
        groups[c]["features"] = np.stack(groups[c]["features"])
    return groups


def _read_embedding(path):
    embedding = torch.load(path)
    return {"indices": embedding["indices"], "features": embedding["features"].numpy()}


def euc_dist(x, y):
    return np.sqrt(np.square(x - y).sum())


def clustering(image_root_dir, embedding_path, indices_path, clustered_save_dir, max_neighbor_dist=7.0):
    """

    :param image_root_dir:
    :param embedding_path:
    :param indices_path: provides class information in this script. each row contains 2 values: path and cls,
        delimited by space. paths provided by this file need to be sorted first to match embedding_path's indices.
    :param clustered_save_dir:
    :param max_neighbor_dist:
    :return:
    """
    image_root_dir = Path(image_root_dir)
    clustered_save_dir = Path(clustered_save_dir)
    embedding = _read_embedding(embedding_path)
    groups = _group_embedding_by_sku(embedding, indices_path)
    for cls, ebd in tqdm(groups.items()):
        indices = np.array(groups[cls]["indices"])
        features = groups[cls]["features"]

        labels = dbscan_gpu(features, max_neighbor_dist=max_neighbor_dist)
        print(f"{len(indices)} -> {labels.max()}")

        cluster_members = defaultdict(list)
        for i, l in enumerate(labels):
            cluster_members[l].append(i)

        for i, (clust_id, members) in enumerate(sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)):

            if len(members) < 2:
                ind = indices[members][0]
                fn = ind.split("/")[-1]
                _save_path = clustered_save_dir / cls / "other-clusters-merged" / fn
                _save_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(image_root_dir / ind, _save_path)
            else:
                for ind in indices[members]:
                    fn = ind.split("/")[-1]
                    _save_path = clustered_save_dir / cls / str(i) / fn
                    _save_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(image_root_dir / ind, _save_path)


if __name__ == '__main__':
    fire.Fire(clustering)
