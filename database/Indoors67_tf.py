import tensorflow as tf
# computed with training data.
MEAN = [94.99279042, 110.04237483, 124.65235427]  # BGR


def num_per_epoche(mode):
    if mode == 'train':
        return 5360
    else:
        return 1340


def multi_crop(img, label, crop_size, crop_num=10):
    print 'img.shape = ', img.get_shape(), '; crop_size:', crop_size
    flipped_image = tf.reverse(img, [1])
    img_shape = img.get_shape().as_list()
    central_ratio = float(crop_size) / img_shape[0]
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


def build_multicrop_input(data_path, dataset='indoors67', color_switch=False):
    with tf.device('/cpu:0'):
        image_size = 256
        crop_size = 224  # size of input images
        if dataset == 'indoors67':
            label_bytes = 1
            label_offset = 0
            num_classes = 67
        else:
            raise ValueError('Not supported dataset %s', dataset)

        depth = 3
        image_bytes = image_size * image_size * depth
        record_bytes = label_bytes + label_offset + image_bytes

        data_files = tf.gfile.Glob(data_path)
        file_queue = tf.train.string_input_producer(data_files, shuffle=False)
        # Read examples from files in the filename queue.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, value = reader.read(file_queue)

        # Convert these examples to dense labels and processed images.
        record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
        label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
        depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]),
                                 [depth, image_size, image_size])
        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

        image -= MEAN

        if color_switch:
            img_b, img_g, img_r = tf.split(axis=2, num_or_size_splits=3, value=image)
            image = tf.cast(tf.concat([img_r, img_g, img_b], 2), dtype=tf.float32)

        images, labels = multi_crop(image, label, crop_size)
        return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)


def build_input(database_server, batch_size, mode,
                examples_per_class=None, dataset='indoors67', blur=True, color_switch=False):

    data_path = '???/train_data_batch_*.bin'

    if mode == 'eval' or mode == 'test' or mode == 'val':
        data_path = '???/test_data_batch_*.bin'

    with tf.device('/cpu:0'):
        image_size = 256
        crop_size = 224  # size of input images
        if dataset == 'indoors67':
            label_bytes = 1
            label_offset = 0
            num_classes = 67
        else:
            raise ValueError('Not supported dataset %s', dataset)

        depth = 3
        image_bytes = image_size * image_size * depth
        record_bytes = label_bytes + label_offset + image_bytes

        data_files = tf.gfile.Glob(data_path)
        print data_files
        file_queue = tf.train.string_input_producer(data_files, shuffle=(mode == 'train'))
        # Read examples from files in the filename queue.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _, value = reader.read(file_queue)

        # Convert these examples to dense labels and processed images.
        record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
        label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
        depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]),
                                 [depth, image_size, image_size])
        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

        if blur:
            image = tf.image.random_brightness(image, max_delta=63. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        image -= MEAN

        if color_switch:
            img_b, img_g, img_r = tf.split(axis=2, num_or_size_splits=3, value=image)
            image = tf.cast(tf.concat([img_r, img_g, img_b], 2), dtype=tf.float32)

        if mode == 'train':
            image = tf.image.random_flip_left_right(image)
            image = tf.random_crop(image, [crop_size, crop_size, 3])

            example_queue = tf.RandomShuffleQueue(
                capacity=16 * batch_size,
                min_after_dequeue=8 * batch_size,
                dtypes=[tf.float32, tf.int32],
                shapes=[[crop_size, crop_size, depth], [1]])
            num_threads = 16
        else:
            # use bigger image for test and validation.
            example_queue = tf.FIFOQueue(
                3 * batch_size,
                dtypes=[tf.float32, tf.int32],
                shapes=[[image_size, image_size, depth], [1]])
            num_threads = 1

        example_enqueue_op = example_queue.enqueue([image, label])
        tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
            example_queue, [example_enqueue_op] * num_threads))

        # Read 'batch' labels + images from the example queue.
        images, labels = example_queue.dequeue_many(batch_size)
        labels = tf.reshape(labels, [batch_size])

        assert len(images.get_shape()) == 4
        assert images.get_shape()[0] == batch_size
        assert images.get_shape()[-1] == 3

        return images, labels
