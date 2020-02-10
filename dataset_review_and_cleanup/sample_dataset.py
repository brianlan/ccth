from pathlib import Path
from collections import defaultdict

import numpy as np
import fire


def _read_indices_and_arrange_into_groups(img_root_dir, indices_path):
    with open(indices_path) as f:
        paths_and_classes = [l.strip().split(" ") for l in f.readlines()]
        if len(paths_and_classes) == 0:
            raise ValueError("indices_path empty.")

    if paths_and_classes[0][0].startswith(str(img_root_dir)):
        paths_and_classes = [[str(Path(p).relative_to(img_root_dir)), c] for p, c in paths_and_classes]

    groups = defaultdict(list)
    for p_and_c in paths_and_classes:
        groups[p_and_c[1]].append(p_and_c[0])
    print(f"Finished reading indices and arrange into groups.")
    return groups


def _sample_images(groups, min_count_to_apply_sampling):
    sampled_paths_and_classes = []
    for cls, paths in groups.items():
        if len(paths) > min_count_to_apply_sampling:
            rnd_sel_paths = np.random.choice(paths, min_count_to_apply_sampling, replace=False).tolist()
            sampled_paths_and_classes.extend([(p, cls) for p in rnd_sel_paths])
        else:
            sampled_paths_and_classes.extend([(p, cls) for p in paths])
    return sampled_paths_and_classes


def _write_indices(paths_and_classes, result_indices_save_path):
    with open(result_indices_save_path, "w") as f:
        f.write("\n".join([" ".join(i) for i in paths_and_classes]))


def sample(img_root_dir, result_indices_save_path, indices_path, min_count_to_apply_sampling=300):
    """

    :param img_root_dir:
    :param result_indices_save_path:
    :param indices_path: plain text file, each row contains 2 values: path (either abs or rel is ok) and class_id,
                         delimited with space.
    :param min_count_to_apply_sampling: if the number of images in a class less or equal than
        `min_count_to_apply_sampling`, it will not get sampled. And for the class contains more images than
        `min_count_to_apply_sampling`, will be sampled exactly `min_count_to_apply_sampling` images.
    :return:
    """
    img_root_dir = Path(img_root_dir)
    result_indices_save_path = Path(result_indices_save_path)
    groups = _read_indices_and_arrange_into_groups(img_root_dir, indices_path)
    sampled_paths_and_classes = _sample_images(groups, min_count_to_apply_sampling)
    _write_indices(sampled_paths_and_classes, result_indices_save_path)


if __name__ == '__main__':
    fire.Fire(sample)
