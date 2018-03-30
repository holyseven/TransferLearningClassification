import tensorflow as tf
import glob


def num_per_epoche(mode, dataset):
    if dataset =='indoors67':
        if mode == 'train':
            return 5360
        else:
            return 1340
    elif dataset == 'dogs120':
        if mode == 'train':
            return 12000
        else:
            return 8580
    elif dataset == 'imagenet':
        if mode == 'train':
            return 1281167
        else:
            return 50000
    else:
        raise ValueError('Not supported dataset %s', dataset)


def multi_crop(img, label, crop_size, image_size, crop_num=10):
    # it is not a best implementation of multiple crops for testing.
    # So use single crop for this moment.
    print 'img.shape = ', image_size, '; crop_size:', crop_size
    flipped_image = tf.reverse(img, [1])
    img_shape = [image_size, image_size]
    central_ratio = float(crop_size) / image_size
    crops = [
        img[:crop_size, :crop_size, :],  # Upper Left
        img[:crop_size, img_shape[1] - crop_size:, :],  # Upper Right
        img[img_shape[0] - crop_size:, :crop_size, :],  # Lower Left
        img[img_shape[0] - crop_size:, img_shape[1] - crop_size:, :],  # Lower Right
        tf.image.central_crop(img, central_ratio),

        flipped_image[:crop_size, :crop_size, :],  # Upper Left
        flipped_image[:crop_size, img_shape[1] - crop_size:, :],  # Upper Right
        flipped_image[img_shape[0] - crop_size:, :crop_size, :],  # Lower Left
        flipped_image[img_shape[0] - crop_size:, img_shape[1] - crop_size:, :],  # Lower Right
        tf.image.central_crop(flipped_image, central_ratio)
    ]

    assert len(crops) == crop_num

    return crops, [label[0] for _ in range(crop_num)]


def simple_central_crop(image, crop_size):
    img_shape = tf.shape(image)
    depth = image.get_shape()[2]

    bbox_h_start = (img_shape[0] - crop_size[0]) / 2
    bbox_w_start = (img_shape[1] - crop_size[1]) / 2

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
        print data_files

        file_queue = tf.train.string_input_producer(data_files, shuffle=(mode == 'train'))

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)

        features = tf.parse_single_example(serialized_example, features=feature_map)

        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        if dataset == 'imagenet':
            label = tf.cast(features['image/class/label'], tf.int32) - 1
        else:
            label = tf.cast(features['image/class/trainid'], tf.int32)

        image = tf.cast(image, tf.float32)

        if resize_image:
            # originally, resize to [image_size, image_size]
            image = tf.image.resize_images(image, [image_size, image_size])
        else:
            # but it is better to keep the scale. L2-SP can have more than 88% precision on Dogs.
            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
            height_smaller_than_width = tf.less_equal(height, width)
            new_shorter_edge = tf.constant(image_size)
            new_height, new_width = tf.cond(
                height_smaller_than_width,
                lambda: (new_shorter_edge, width * new_shorter_edge / height),
                lambda: (height * new_shorter_edge / width, new_shorter_edge))
            image = tf.image.resize_images(image, [new_height, new_width])

        if blur:
            image = tf.image.random_brightness(image, max_delta=63. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        image -= IMG_MEAN

        # rgb (in db) -> bgr, depends on the pre-trained model.
        # if model transferred from caffe model, use bgr; else, use rgb.
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
                print 'use multiple crops and the test batch size is not used.'
                batch_images, batch_labels = multi_crop(image, label, crop_size, image_size)
                batch_images = tf.convert_to_tensor(batch_images)
                batch_labels = tf.convert_to_tensor(batch_labels)
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
