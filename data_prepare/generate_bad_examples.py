from pathlib import Path

import numpy as np
import fire
from PIL import Image

from src.crop_utils import random_crop


def generate(src_img_root_dir, dst_img_dir, sample_rate=1.0):
    """

    :param src_img_root_dir:
    :param dst_img_dir:
    :param sample_rate: if less than 1.0, only a portion of the source images will be used to generate bad examples.
    :return:
    """
    Path(dst_img_dir).mkdir(parents=True, exist_ok=True)
    im_paths = list(Path(src_img_root_dir).rglob("*.jpg"))
    im_paths = np.random.choice(im_paths, int(round(len(im_paths) * sample_rate)))
    for im_path in im_paths:
        im = Image.open(im_path)
        if im.size[0] / im.size[1] > 0.625:  # currently, we only apply the crop to slim long products.
            continue
        cropped_im = random_crop(im)
        cropped_im.save(str(Path(dst_img_dir) / im_path.name))


if __name__ == "__main__":
    fire.Fire(generate)
