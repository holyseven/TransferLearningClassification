from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob


def num_per_epoche(mode, dataset):
    if dataset == 'indoors67':
        if mode == 'train':
            return 5360
        else:
            return 1340
    elif dataset == 'dogs120':
        if mode == 'train':
            return 12000
        else:
            return 8580  # 2*2*3*5*11*13
    elif dataset == 'foods101':
        if mode == 'train':
            return 75750
        else:
            return 25250
    elif dataset == 'caltech256':
        if mode == 'train':
            return 15420  # 60*257 = 15420
        else:  # rest: 10047, train: 60*257, test: 20 *257
            return 5140
    elif dataset == 'places365':
        if mode == 'train':
            return 1803460
        else:
            return 36500
    elif dataset == 'imagenet':
        if mode == 'train':
            return 1281167
        else:
            return 50000
    else:
        raise ValueError('Not supported dataset %s', dataset)


def multi_crop(img, label, crop_size, image_size, crop_num=10):
    # it is not a best implementation of multiple crops for testing.

    def central_crop(img, crop_size):
        img_shape = tf.shape(img)
        depth = img.get_shape()[2]
        img_h = tf.to_double(img_shape[0])
        img_w = tf.to_double(img_shape[1])
        bbox_h_start = tf.to_int32((img_h - crop_size) / 2)
        bbox_w_start = tf.to_int32((img_w - crop_size) / 2)

        bbox_begin = tf.stack([bbox_h_start, bbox_w_start, 0])
        bbox_size = tf.stack([crop_size, crop_size, -1])
        image = tf.slice(img, bbox_begin, bbox_size)

        # The first two dimensions are dynamic and unknown.
        image.set_shape([crop_size, crop_size, depth])
        return image

    print('img.shape = ', image_size, '; crop_size:', crop_size)
    flipped_image = tf.reverse(img, [1])
    img_shape = tf.shape(img)
    crops = [
        img[:crop_size, :crop_size, :],  # Upper Left
        img[:crop_size, img_shape[1] - crop_size:, :],  # Upper Right
        img[img_shape[0] - crop_size:, :crop_size, :],  # Lower Left
        img[img_shape[0] - crop_size:, img_shape[1] - crop_size:, :],  # Lower Right
        central_crop(img, crop_size),

        flipped_image[:crop_size, :crop_size, :],  # Upper Left
        flipped_image[:crop_size, img_shape[1] - crop_size:, :],  # Upper Right
        flipped_image[img_shape[0] - crop_size:, :crop_size, :],  # Lower Left
        flipped_image[img_shape[0] - crop_size:, img_shape[1] - crop_size:, :],  # Lower Right
        central_crop(flipped_image, crop_size)
    ]

    assert len(crops) == crop_num

    return crops, [label[0] for _ in range(crop_num)]


def simple_central_crop(image, crop_size):
    img_shape = tf.shape(image)
    depth = image.get_shape()[2]

    bbox_h_start = (img_shape[0] - crop_size[0]) // 2
    bbox_w_start = (img_shape[1] - crop_size[1]) // 2

    bbox_begin = tf.stack([bbox_h_start, bbox_w_start, 0])
    bbox_size = tf.stack([crop_size[0], crop_size[1], -1])
    image = tf.slice(image, bbox_begin, bbox_size)

    image.set_shape([crop_size[0], crop_size[1], depth])
    return image


def build_input(batch_size, mode, dataset='dogs120', blur=True, color_switch=False, resize_image=True,
                resize_image_size=256, crop_size=224, examples_per_class=None, multicrops_for_eval=False):
    with tf.device('/cpu:0'):
        image_size = resize_image_size

        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
            'image/class/trainid': tf.FixedLenFeature([1], dtype=tf.int64,
                                                      default_value=-1),
        }

        if dataset == 'indoors67':
            num_classes = 67
            # computed with training data.
            IMG_MEAN = [123.55967712, 110.04705048, 93.7353363]  # RGB [124.65235427, 110.04237483, 94.99279042]
            data_path = '../create_databases/tfRecords-Indoors/Train-*'
            if 'val' in mode or 'test' in mode:
                data_path = '../create_databases/tfRecords-Indoors/Test-*'
        elif dataset == 'dogs120':
            num_classes = 120
            # computed with training data.
            IMG_MEAN = [120.40182495, 115.22045135, 98.45511627]  # RGB [121.49871251, 115.17340629, 99.73959828]
            data_path = '../create_databases/tfRecords-Dogs/train-*'
            if 'val' in mode or 'test' in mode:
                data_path = '../create_databases/tfRecords-Dogs/test-*'
        elif dataset == 'caltech256':
            num_classes = 257
            # computed with training data.
            IMG_MEAN = [140.48295593, 135.94039917, 127.60546112]  # RGB [140.48295593, 135.94039917, 127.60546112]
            data_path = '../create_databases/tfRecords-Caltech/train-*'
            if 'val' in mode or 'test' in mode:
                data_path = '../create_databases/tfRecords-Caltech/test-*'  # test.
        elif dataset == 'foods101':
            num_classes = 101
            # computed with training data.
            IMG_MEAN = [137.87016502, 113.09658086, 86.3819538]  # RGB [137.87016502, 113.09658086, 86.3819538]
            data_path = '../create_databases/tfRecords-Foods/train-*'
            if 'val' in mode or 'test' in mode:
                data_path = '../create_databases/tfRecords-Foods/test-*'
        elif dataset == 'places365':
            num_classes = 365
            IMG_MEAN = [115.59942627, 112.52274323, 102.81996155]  # RGB
            data_path = '../create_databases/tfRecords-Places/train*'
            if 'val' in mode:
                data_path = '../create_databases/tfRecords-Places/val*'
        elif dataset == 'imagenet':
            num_classes = 1000
            IMG_MEAN = [123.68, 116.779, 103.939]  # RGB
            data_path = '../create_databases/tfRecords-ImageNet/train-*'
            if 'val' in mode:
                data_path = '../create_databases/tfRecords-ImageNet/validation-*'
            feature_map = {
                'image/height': tf.FixedLenFeature([1], dtype=tf.int64,
                                                   default_value=-1),
                'image/width': tf.FixedLenFeature([1], dtype=tf.int64,
                                                  default_value=-1),
                'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                    default_value=''),
                'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                        default_value=-1),
                'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                       default_value=''),
            }
        else:
            raise ValueError('Not supported dataset %s', dataset)

        data_files = glob.glob(data_path)
        if dataset == 'caltech256' and mode == 'train':
            num_files = int(examples_per_class // 5)
            data_files = sorted(data_files)[0:num_files]
        print(data_files)
        assert len(data_files) > 0, 'No database is found.'

        file_queue = tf.train.string_input_producer(data_files, shuffle=(mode == 'train'))

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)
        features = tf.parse_single_example(serialized_example, features=feature_map)
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        image = tf.cast(image, tf.float32)

        if dataset == 'imagenet':
            label = tf.cast(features['image/class/label'], tf.int32) - 1
        else:
            label = tf.cast(features['image/class/trainid'], tf.int32)

        if resize_image:
            # originally, resize to [image_size, image_size]
            image = tf.image.resize_images(image, [image_size, image_size])
        else:
            # but it is better to keep the scale.
            # L2-SP can have more than 88% precision on Dogs, with a pre-trained resnet-101 model.
            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
            height_smaller_than_width = tf.less_equal(height, width)
            new_shorter_edge = tf.constant(image_size)
            new_height, new_width = tf.cond(
                height_smaller_than_width,
                lambda: (new_shorter_edge, width * new_shorter_edge // height),
                lambda: (height * new_shorter_edge // width, new_shorter_edge))
            image = tf.image.resize_images(image, [new_height, new_width])

        if blur:
            image = tf.image.random_brightness(image, max_delta=63. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        # image is an RGB image. So subtract an RGB value.
        image -= IMG_MEAN

        # color_switch: rgb (in db) -> bgr, depends on the pre-trained model.
        # if model is transferred from caffe model, use bgr; else, use rgb.
        if color_switch:
            img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=image)
            image = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)

        if mode == 'train':
            image = tf.image.random_flip_left_right(image)
            image = tf.random_crop(image, [crop_size, crop_size, 3])
            num_threads = 8
            batch_images, batch_labels = tf.train.shuffle_batch([image, label[0]], batch_size=batch_size,
                                                                capacity=16 * batch_size,
                                                                num_threads=num_threads,
                                                                min_after_dequeue=8 * batch_size)
        else:
            if multicrops_for_eval:
                print('use multiple crops and the test batch size is not used.')
                batch_images, batch_labels = multi_crop(image, label, crop_size, image_size)
                batch_images = tf.convert_to_tensor(batch_images)
                batch_labels = tf.convert_to_tensor(batch_labels)
                batch_size = batch_images.get_shape()[0]
            else:
                image = simple_central_crop(image, [crop_size, crop_size])
                num_threads = 4
                batch_images, batch_labels = tf.train.batch([image, label[0]], batch_size=batch_size,
                                                            capacity=16 * batch_size,
                                                            num_threads=num_threads)

        assert len(batch_images.get_shape()) == 4
        assert batch_images.get_shape()[0] == batch_size
        assert batch_images.get_shape()[-1] == 3

        return batch_images, batch_labels, num_classes
