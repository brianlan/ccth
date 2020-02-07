from ..crop_utils import convert_to_crop_param


def test_convert_to_crop_param_1():
    assert convert_to_crop_param([0.0, 0.0, 0.25, 1.0], (380, 630)) == [0, 0, 630, 95]
    assert convert_to_crop_param([0.75, 0.0, 0.25, 1.0], (380, 630)) == [0, 285, 630, 95]
    assert convert_to_crop_param([0.0, 0.0, 1.0, 0.35], (380, 630)) == [0, 0, 220, 380]
    assert convert_to_crop_param([0.0, 0.65, 1.0, 0.35], (380, 630)) == [409, 0, 220, 380]


def test_convert_to_crop_param_2():
    assert convert_to_crop_param([1.0, 0.0, 0.25, 1.0], (3, 11)) == [0, 2, 11, 1]
    assert convert_to_crop_param([0.85, 0.0, 0.25, 1.0], (3, 11)) == [0, 2, 11, 1]
    assert convert_to_crop_param([0.0, 0.0, 1.0, 0.35], (3, 11)) == [0, 0, 4, 3]
    assert convert_to_crop_param([0.0, 0.65, 1.0, 0.35], (3, 11)) == [7, 0, 4, 3]
