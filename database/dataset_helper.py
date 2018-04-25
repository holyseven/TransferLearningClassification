from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def multicrops_np_image(one_image, crop_image_size=224):
    # print(one_image.shape)  # [height, width, 3]
    height, width, _ = one_image.shape

    dif_height = height - crop_image_size
    dif_width = width - crop_image_size

    if dif_height < 0 or dif_width < 0:
        one_image = numpy_pad_image(one_image, dif_height, dif_width)
        height, width = one_image.shape[0], one_image.shape[1]

    mirror_one_image = one_image[:, ::-1]

    heights = decide_intersection(height, crop_image_size)
    widths = decide_intersection(width, crop_image_size)

    split_crops = []
    for height in heights:
        for width in widths:
            image_crop = one_image[height:height + crop_image_size, width:width + crop_image_size]
            mirror_crop = mirror_one_image[height:height + crop_image_size, width:width + crop_image_size]
            split_crops.append(image_crop[np.newaxis, :])
            split_crops.append(mirror_crop[np.newaxis, :])

    split_crops = np.concatenate(split_crops, axis=0)  # (n, crop_image_size, crop_image_size, 3)

    return split_crops


def numpy_crop_image(image, dif_height, dif_width):
    # (height, width, channel)
    assert len(image.shape) == 3
    if dif_height < 0:
        if dif_height % 2 == 0:
            pad_before_h = - dif_height // 2
            pad_after_h = dif_height // 2
        else:
            pad_before_h = - dif_height // 2
            pad_after_h = dif_height // 2
        image = image[pad_before_h:pad_after_h]

    if dif_width < 0:
        if dif_width % 2 == 0:
            pad_before_w = - dif_width // 2
            pad_after_w = dif_width // 2
        else:
            pad_before_w = - dif_width // 2
            pad_after_w = dif_width // 2
        image = image[:, pad_before_w:pad_after_w]

    return image


def decide_intersection(total_length, crop_length):
    stride = crop_length * 1 // 3
    times = (total_length - crop_length) // stride + 1
    cropped_starting = []
    for i in range(times):
        cropped_starting.append(stride*i)
    if total_length - cropped_starting[-1] > crop_length:
        cropped_starting.append(total_length - crop_length)
    return cropped_starting


def numpy_pad_image(image, total_padding_h, total_padding_w, image_padding_value=0):
    # (height, width, channel)
    assert len(image.shape) == 3
    pad_before_w = pad_after_w = 0
    pad_before_h = pad_after_h = 0
    if total_padding_h < 0:
        if total_padding_h % 2 == 0:
            pad_before_h = pad_after_h = - total_padding_h // 2
        else:
            pad_before_h = - total_padding_h // 2
            pad_after_h = - total_padding_h // 2 + 1
    if total_padding_w < 0:
        if total_padding_w % 2 == 0:
            pad_before_w = pad_after_w = - total_padding_w // 2
        else:
            pad_before_w = - total_padding_w // 2
            pad_after_w = - total_padding_w // 2 + 1
    image_crop = np.pad(image,
                        ((pad_before_h, pad_after_h), (pad_before_w, pad_after_w), (0, 0)),
                        mode='constant', constant_values=image_padding_value)
    return image_crop
