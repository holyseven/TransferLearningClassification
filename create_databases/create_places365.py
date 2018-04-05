
from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_integer('val_shards', 4,
                            'Number of shards in test TFRecord files.')
tf.app.flags.DEFINE_string('output_directory', './tfRecords-Places/',
                           'Output data directory')
tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, trainid):
    """Build an Example proto for an example.
    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
      human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
      bbox: list of bounding boxes; each box is a list of integers
        specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
        the same label as the image label.
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/class/trainid': _int64_feature(trainid),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


def _is_bmp(filename):
    blacklist = ['laundromat/Laundry_Room.bmp',
                 'waitingroom/Bistro_3.BMP',
                 'kindergarden/classroom_north.bmp']
    return filename.split('Images/')[-1] in blacklist


class ImageCoder(object):
    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes RGB JPEG data.
        self._input_data = tf.placeholder(dtype=tf.string)
        self._image_data = tf.image.decode_jpeg(self._input_data, channels=3)
        self._encoded_data = tf.image.encode_jpeg(self._image_data, format='rgb', quality=100)

        self._bmp_data = tf.image.decode_bmp(self._input_data, channels=3)
        self._bmp_to_jpeg = tf.image.encode_jpeg(self._bmp_data, format='rgb', quality=100)

    def re_encode_jpeg(self, image_data):
        # since tf1.2, decode_jpeg can decode JPEGs, PNGs and non-animated GIFs; so for compatibility,
        # re-encoding all of three to jpegs for version < 1.2.
        return self._sess.run(self._encoded_data,
                              feed_dict={self._input_data: image_data})

    def bmp_to_jpeg(self, image_data):
        return self._sess.run(self._bmp_to_jpeg,
                              feed_dict={self._input_data: image_data})


def _process_image(filename, coder):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()

    if _is_bmp(filename):
        print 'Converting BMP to JPEG for %s' % filename
        image_data = coder.bmp_to_jpeg(image_data)

    return image_data


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, labels, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]

            image_buffer = _process_image(filename, coder)

            example = _convert_to_example(image_buffer, label)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, labels, num_shards):
    """Process and save list of images as TFRecord of Example protos.
    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir, subdir='val_256'):
    imageListFilename = data_dir + 'filelist_places365-standard/places365_val.txt'

    file_list = open(imageListFilename).read().split('\n')
    # file_list.pop()

    filenames = []
    labels = []

    for filename in file_list:
        label = int(filename.split(' ')[-1])
        labels.append(label)
        filenames.append(data_dir + '/val_256/' + filename.split(' ')[0])

    return filenames, labels


def _process_dataset(name, directory, num_shards):
    """Process a complete data set and save it as a TFRecord.
    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      synset_to_human: dict of synset to human labels, e.g.,
        'n02119022' --> 'red fox, Vulpes vulpes'
      image_to_bboxes: dictionary mapping image file names to a list of
        bounding boxes. This list contains 0+ bounding boxes.
    """
    filenames, labels = _find_image_files(directory)
    _process_image_files(name, filenames, labels, num_shards)


def main(unused_argv):
    assert not FLAGS.val_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')

    if os.path.exists(FLAGS.output_directory) is not True:
        os.mkdir(FLAGS.output_directory)

    dir_name = '/home/jacques/workspace/database/Places365/'

    # Run it!
    _process_dataset('val_256', dir_name, FLAGS.val_shards)


if __name__ == '__main__':
    tf.app.run()
