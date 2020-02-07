from math import floor

import torchvision.transforms.functional as F
import numpy as np


def random_crop(img):
    chosen_param = get_percent_param()
    crop_param = convert_to_crop_param(chosen_param, img.size)
    return F.crop(img, *crop_param)


def get_percent_param():
    # each choices is a 4 value tuple: [left, top, width, height], all values should be given in percent
    param_choices = [[0.0, 0.0, 0.25, 1.0], [0.75, 0.0, 0.25, 1.0], [0.0, 0.0, 1.0, 0.35], [0.0, 0.65, 1.0, 0.35]]
    chosen_idx = np.random.randint(len(param_choices))
    return param_choices[chosen_idx]


def convert_to_crop_param(percent_param, im_size):
    return [
        max(0, min(int(floor(percent_param[1] * im_size[1])), im_size[1] - 1)),
        max(0, min(int(floor(percent_param[0] * im_size[0])), im_size[0] - 1)),
        int(round(percent_param[3] * im_size[1])),
        int(round(percent_param[2] * im_size[0])),
    ]
