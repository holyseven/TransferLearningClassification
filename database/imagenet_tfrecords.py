import tensorflow as tf
MEAN = [123.68, 116.779, 103.939]  # RGB


def num_per_epoche(mode):
    if mode == 'train':
        return 1281167
    else:
        return 50000


def central_crop(image, crop_size):
    img_shape = tf.shape(image)
    depth = image.get_shape()[2]

    bbox_h_start = (img_shape[0] - crop_size[0]) / 2
    bbox_w_start = (img_shape[1] - crop_size[1]) / 2

    bbox_begin = tf.stack([bbox_h_start, bbox_w_start, 0])
    bbox_size = tf.stack([crop_size[0], crop_size[1], -1])
    image = tf.slice(image, bbox_begin, bbox_size)

    image.set_shape([crop_size[0], crop_size[1], depth])
    return image


def build_input(database_server, batch_size, mode, dataset='imagenet', blur=True, color_switch=False):

    with tf.device('/cpu:0'):
        image_size = 256
        crop_size = 224  # size of input images
        depth = 3
        if dataset == 'imagenet':
            num_classes = 1000
        else:
            raise ValueError('Not supported dataset %s', dataset)

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

        data_path = '???/train-*-of-00100'

        if mode == 'eval' or mode == 'test' or mode == 'val':
            data_path = '???/validation-*-of-00005'

        data_files = tf.gfile.Glob(data_path)
        print data_files

        file_queue = tf.train.string_input_producer(data_files, shuffle=(mode == 'train'))

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)

        features = tf.parse_single_example(serialized_example, features=feature_map)

        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)

        label = tf.cast(features['image/class/label'], tf.int32)
        height = tf.cast(features['image/height'], tf.int32)[0]
        width = tf.cast(features['image/width'], tf.int32)[0]

        image = tf.cast(image, tf.float32)

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

        image -= MEAN

        if color_switch:
            img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=image)
            image = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)

        if mode == 'train':
            image = tf.image.random_flip_left_right(image)
            image = tf.random_crop(image, [crop_size, crop_size, 3])
            num_threads = 8
            batch_images, batch_labels = tf.train.shuffle_batch([image, label[0] - 1], batch_size=batch_size,
                                                                capacity=16 * batch_size,
                                                                num_threads=num_threads,
                                                                min_after_dequeue=8 * batch_size)
        else:
            image = central_crop(image, [crop_size, crop_size])
            num_threads = 4
            batch_images, batch_labels = tf.train.batch([image, label[0] - 1], batch_size=batch_size,
                                                        capacity=16 * batch_size,
                                                        num_threads=num_threads)

        assert len(batch_images.get_shape()) == 4
        assert batch_images.get_shape()[0] == batch_size
        assert batch_images.get_shape()[-1] == 3

        return batch_images, batch_labels


