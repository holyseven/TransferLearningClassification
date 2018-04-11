
from datetime import datetime
import os
import random
import sys
import threading
import glob

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_integer('train_shards', 12,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('test_shards', 4,
                            'Number of shards in test TFRecord files.')
tf.app.flags.DEFINE_integer('rest_shards', 4,
                            'Number of shards in rest TFRecord files.')
tf.app.flags.DEFINE_string('output_directory', './tfRecords-Caltech/',
                           'Output data directory')
tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS


def mapping_name_to_label(image_dir):
    a = dict()
    index = 0
    for name in sorted(glob.glob(image_dir + '/*')):
        a[name.split('/')[-1]] = index
        index += 1

    return a


def mapping_label_to_name(image_dir):
    a = []
    for name in sorted(glob.glob(image_dir + '/*')):
        a.append(name.split('/')[-1])

    return a


def generate_image_filenames_and_label(img_index, image_dir):
    """

    :param img_index: five ints for indicating images.
    :return:
    """

    assert len(img_index) == 5
    map_lable_to_name = mapping_label_to_name(image_dir)
    # print map_lable_to_name
    filenames = []
    lables = []
    for i in range(257):
        index_in_dir = i + 1
        index_str = '%03d'%index_in_dir

        for j in img_index:
            img_j_str = '%04d'%j
            filenames.append(image_dir + '/' + map_lable_to_name[i] + '/' + index_str + '_' + img_j_str + '.jpg')
            lables.append(i)

    assert len(filenames) == len(lables)
    index = range(len(filenames))
    np.random.shuffle(index)

    filenames_output = []
    lables_output = []
    for n in index:
        filenames_output.append(filenames[n])
        lables_output.append(lables[n])

    return filenames_output, lables_output


def all_files_beyond_80(image_dir):
    index = range(81, 900, 1)
    map_lable_to_name = mapping_label_to_name(image_dir)
    filenames = []
    lables = []
    for i in range(257):
        index_in_dir = i + 1
        index_str = '%03d'%index_in_dir

        for j in index:
            img_j_str = '%04d'%j
            filename = image_dir + '/' + map_lable_to_name[i] + '/' + index_str + '_' + img_j_str + '.jpg'
            if os.path.isfile(filename) is False:
                break
            filenames.append(filename)
            lables.append(i)

    return filenames, lables


def generate_lists(image_dir, subdir):
    """
    80 images per class in total and 256 classes. Generate a list of lists. Each list contains 256*5 images.
    :return: a list of lists. Each list contains 256*5 images.
    """

    # last 20 images are for test.
    index = range(1, 81, 1)
    # np.random.shuffle(index)

    list_of_filenames = []
    list_of_lables = []
    if subdir == 'train':
        # for training, 12 lists
        for i in range(12):
            index_for_one_list = index[i*5:(i+1)*5]
            filenames, lables = generate_image_filenames_and_label(index_for_one_list, image_dir)
            list_of_filenames.extend(filenames)
            list_of_lables.extend(lables)
    elif subdir == 'test':
        # for test, 4 lists
        for i in range(12, 16, 1):
            index_for_one_list = index[i*5:(i+1)*5]
            filenames, lables = generate_image_filenames_and_label(index_for_one_list, image_dir)
            list_of_filenames.extend(filenames)
            list_of_lables.extend(lables)
    elif subdir == 'rest':
        return all_files_beyond_80(image_dir)
    else:
        return None

    return list_of_filenames, list_of_lables


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


def _convert_to_example(image_buffer, trainid, filename):
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
        'image/encoded': _bytes_feature(image_buffer),
        'image/filename': _bytes_feature(filename)}))
    return example


def _is2convert(filename):
    blacklist = [
        '043.coin/043_0023.jpg',
        '252.car-side-101/252_0023.jpg',
        '165.pram/165_0025.jpg',
        '230.trilobite-101/230_0025.jpg',
        '147.mushroom/147_0022.jpg',
        '252.car-side-101/252_0022.jpg',
        '252.car-side-101/252_0025.jpg',
        '111.house-fly/111_0024.jpg',
        '225.tower-pisa/225_0025.jpg',
        '229.tricycle/229_0023.jpg',
        '252.car-side-101/252_0024.jpg',
        '094.guitar-pick/094_0021.jpg',
        '144.minotaur/144_0023.jpg',
        '252.car-side-101/252_0021.jpg',
        '044.comet/044_0021.jpg',
        '247.xylophone/247_0022.jpg',
        '184.sheet-music/184_0023.jpg',
        '140.menorah-101/140_0022.jpg',
        '014.blimp/014_0022.jpg',
        '057.dolphin-101/057_0025.jpg',
        '222.tombstone/222_0032.jpg',
        '252.car-side-101/252_0033.jpg',
        '028.camel/028_0031.jpg',
        '174.rotary-phone/174_0033.jpg',
        '104.homer-simpson/104_0034.jpg',
        '089.goose/089_0033.jpg',
        '252.car-side-101/252_0031.jpg',
        '252.car-side-101/252_0032.jpg',
        '165.pram/165_0032.jpg',
        '252.car-side-101/252_0035.jpg',
        '082.galaxy/082_0032.jpg',
        '163.playing-card/163_0035.jpg',
        '234.tweezer/234_0031.jpg',
        '140.menorah-101/140_0033.jpg',
        '184.sheet-music/184_0032.jpg',
        '172.revolver-101/172_0032.jpg',
        '111.house-fly/111_0031.jpg',
        '148.mussels/148_0034.jpg',
        '252.car-side-101/252_0034.jpg',
        '252.car-side-101/252_0059.jpg',
        '082.galaxy/082_0057.jpg',
        '075.floppy-disk/075_0058.jpg',
        '044.comet/044_0057.jpg',
        '140.menorah-101/140_0060.jpg',
        '020.brain-101/020_0059.jpg',
        '172.revolver-101/172_0056.jpg',
        '153.palm-pilot/153_0057.jpg',
        '104.homer-simpson/104_0056.jpg',
        '252.car-side-101/252_0056.jpg',
        '252.car-side-101/252_0058.jpg',
        '172.revolver-101/172_0059.jpg',
        '127.laptop-101/127_0058.jpg',
        '044.comet/044_0058.jpg',
        '252.car-side-101/252_0060.jpg',
        '252.car-side-101/252_0057.jpg',
        '034.centipede/034_0059.jpg',
        '252.car-side-101/252_0017.jpg',
        '225.tower-pisa/225_0020.jpg',
        '234.tweezer/234_0020.jpg',
        '159.people/159_0019.jpg',
        '168.raccoon/168_0018.jpg',
        '184.sheet-music/184_0017.jpg',
        '111.house-fly/111_0019.jpg',
        '144.minotaur/144_0016.jpg',
        '214.teepee/214_0019.jpg',
        '111.house-fly/111_0018.jpg',
        '184.sheet-music/184_0019.jpg',
        '140.menorah-101/140_0019.jpg',
        '252.car-side-101/252_0016.jpg',
        '252.car-side-101/252_0020.jpg',
        '013.birdbath/013_0020.jpg',
        '252.car-side-101/252_0018.jpg',
        '028.camel/028_0020.jpg',
        '226.traffic-light/226_0020.jpg',
        '252.car-side-101/252_0019.jpg',
        '001.ak47/001_0016.jpg',
        '030.canoe/030_0009.jpg',
        '252.car-side-101/252_0008.jpg',
        '252.car-side-101/252_0007.jpg',
        '061.dumb-bell/061_0008.jpg',
        '184.sheet-music/184_0010.jpg',
        '155.paperclip/155_0006.jpg',
        '020.brain-101/020_0009.jpg',
        '244.wheelbarrow/244_0010.jpg',
        '212.teapot/212_0008.jpg',
        '252.car-side-101/252_0009.jpg',
        '111.house-fly/111_0009.jpg',
        '006.basketball-hoop/006_0007.jpg',
        '214.teepee/214_0006.jpg',
        '230.trilobite-101/230_0009.jpg',
        '210.syringe/210_0010.jpg',
        '165.pram/165_0006.jpg',
        '155.paperclip/155_0008.jpg',
        '252.car-side-101/252_0010.jpg',
        '252.car-side-101/252_0006.jpg',
        '044.comet/044_0006.jpg',
        '022.buddha-101/022_0009.jpg',
        '044.comet/044_0041.jpg',
        '036.chandelier-101/036_0045.jpg',
        '052.crab-101/052_0045.jpg',
        '172.revolver-101/172_0042.jpg',
        '252.car-side-101/252_0042.jpg',
        '252.car-side-101/252_0044.jpg',
        '252.car-side-101/252_0045.jpg',
        '031.car-tire/031_0042.jpg',
        '020.brain-101/020_0041.jpg',
        '255.tennis-shoes/255_0045.jpg',
        '252.car-side-101/252_0043.jpg',
        '172.revolver-101/172_0041.jpg',
        '214.teepee/214_0043.jpg',
        '193.soccer-ball/193_0041.jpg',
        '252.car-side-101/252_0041.jpg',
        '184.sheet-music/184_0045.jpg',
        '172.revolver-101/172_0043.jpg',
        '184.sheet-music/184_0044.jpg',
        '234.tweezer/234_0044.jpg',
        '201.starfish-101/201_0044.jpg',
        '043.coin/043_0042.jpg',
        '184.sheet-music/184_0043.jpg',
        '234.tweezer/234_0043.jpg',
        '244.wheelbarrow/244_0041.jpg',
        '006.basketball-hoop/006_0004.jpg',
        '252.car-side-101/252_0002.jpg',
        '094.guitar-pick/094_0001.jpg',
        '252.car-side-101/252_0004.jpg',
        '159.people/159_0003.jpg',
        '230.trilobite-101/230_0002.jpg',
        '184.sheet-music/184_0001.jpg',
        '167.pyramid/167_0003.jpg',
        '252.car-side-101/252_0005.jpg',
        '252.car-side-101/252_0001.jpg',
        '171.refrigerator/171_0003.jpg',
        '006.basketball-hoop/006_0005.jpg',
        '234.tweezer/234_0001.jpg',
        '184.sheet-music/184_0005.jpg',
        '199.spoon/199_0003.jpg',
        '131.lightbulb/131_0003.jpg',
        '123.ketch-101/123_0002.jpg',
        '064.elephant-101/064_0005.jpg',
        '252.car-side-101/252_0003.jpg',
        '255.tennis-shoes/255_0047.jpg',
        '060.duck/060_0049.jpg',
        '040.cockroach/040_0050.jpg',
        '044.comet/044_0049.jpg',
        '201.starfish-101/201_0047.jpg',
        '064.elephant-101/064_0050.jpg',
        '252.car-side-101/252_0048.jpg',
        '184.sheet-music/184_0048.jpg',
        '252.car-side-101/252_0046.jpg',
        '219.theodolite/219_0048.jpg',
        '252.car-side-101/252_0047.jpg',
        '252.car-side-101/252_0050.jpg',
        '062.eiffel-tower/062_0047.jpg',
        '252.car-side-101/252_0049.jpg',
        '184.sheet-music/184_0047.jpg',
        '058.doorknob/058_0046.jpg',
        '115.ice-cream-cone/115_0046.jpg',
        '032.cartman/032_0047.jpg',
        '177.saturn/177_0047.jpg',
        '022.buddha-101/022_0047.jpg',
        '040.cockroach/040_0049.jpg',
        '177.saturn/177_0051.jpg',
        '044.comet/044_0053.jpg',
        '082.galaxy/082_0055.jpg',
        '058.doorknob/058_0053.jpg',
        '252.car-side-101/252_0051.jpg',
        '032.cartman/032_0055.jpg',
        '094.guitar-pick/094_0052.jpg',
        '038.chimp/038_0053.jpg',
        '140.menorah-101/140_0051.jpg',
        '144.minotaur/144_0054.jpg',
        '020.brain-101/020_0053.jpg',
        '064.elephant-101/064_0051.jpg',
        '094.guitar-pick/094_0051.jpg',
        '252.car-side-101/252_0052.jpg',
        '191.sneaker/191_0053.jpg',
        '234.tweezer/234_0055.jpg',
        '094.guitar-pick/094_0055.jpg',
        '020.brain-101/020_0054.jpg',
        '252.car-side-101/252_0053.jpg',
        '252.car-side-101/252_0054.jpg',
        '045.computer-keyboard/045_0053.jpg',
        '044.comet/044_0052.jpg',
        '252.car-side-101/252_0055.jpg',
        '155.paperclip/155_0039.jpg',
        '082.galaxy/082_0040.jpg',
        '252.car-side-101/252_0038.jpg',
        '172.revolver-101/172_0037.jpg',
        '197.speed-boat/197_0037.jpg',
        '252.car-side-101/252_0040.jpg',
        '252.car-side-101/252_0037.jpg',
        '036.chandelier-101/036_0039.jpg',
        '111.house-fly/111_0040.jpg',
        '139.megaphone/139_0036.jpg',
        '219.theodolite/219_0038.jpg',
        '144.minotaur/144_0036.jpg',
        '044.comet/044_0038.jpg',
        '020.brain-101/020_0040.jpg',
        '244.wheelbarrow/244_0038.jpg',
        '184.sheet-music/184_0040.jpg',
        '044.comet/044_0036.jpg',
        '214.teepee/214_0039.jpg',
        '252.car-side-101/252_0039.jpg',
        '229.tricycle/229_0036.jpg',
        '252.car-side-101/252_0036.jpg',
        '184.sheet-music/184_0039.jpg',
        '195.soda-can/195_0039.jpg',
        '252.car-side-101/252_0028.jpg',
        '022.buddha-101/022_0028.jpg',
        '167.pyramid/167_0029.jpg',
        '222.tombstone/222_0027.jpg',
        '252.car-side-101/252_0027.jpg',
        '252.car-side-101/252_0029.jpg',
        '110.hourglass/110_0030.jpg',
        '201.starfish-101/201_0027.jpg',
        '140.menorah-101/140_0026.jpg',
        '252.car-side-101/252_0026.jpg',
        '252.car-side-101/252_0030.jpg',
        '252.car-side-101/252_0014.jpg',
        '044.comet/044_0011.jpg',
        '223.top-hat/223_0015.jpg',
        '252.car-side-101/252_0013.jpg',
        '058.doorknob/058_0011.jpg',
        '044.comet/044_0013.jpg',
        '252.car-side-101/252_0012.jpg',
        '020.brain-101/020_0013.jpg',
        '184.sheet-music/184_0012.jpg',
        '020.brain-101/020_0012.jpg',
        '020.brain-101/020_0011.jpg',
        '252.car-side-101/252_0011.jpg',
        '155.paperclip/155_0015.jpg',
        '020.brain-101/020_0014.jpg',
        '252.car-side-101/252_0015.jpg',
        '184.sheet-music/184_0011.jpg',
        '083.gas-pump/083_0014.jpg',
        '123.ketch-101/123_0079.jpg',
        '020.brain-101/020_0076.jpg',
        '155.paperclip/155_0080.jpg',
        '252.car-side-101/252_0076.jpg',
        '066.ewer-101/066_0078.jpg',
        '252.car-side-101/252_0078.jpg',
        '244.wheelbarrow/244_0079.jpg',
        '066.ewer-101/066_0077.jpg',
        '172.revolver-101/172_0078.jpg',
        '226.traffic-light/226_0080.jpg',
        '191.sneaker/191_0076.jpg',
        '082.galaxy/082_0080.jpg',
        '153.palm-pilot/153_0080.jpg',
        '252.car-side-101/252_0077.jpg',
        '062.eiffel-tower/062_0077.jpg',
        '222.tombstone/222_0078.jpg',
        '252.car-side-101/252_0080.jpg',
        '252.car-side-101/252_0079.jpg',
        '052.crab-101/052_0080.jpg',
        '063.electric-guitar-101/063_0080.jpg',
        '020.brain-101/020_0080.jpg',
        '155.paperclip/155_0067.jpg',
        '252.car-side-101/252_0069.jpg',
        '177.saturn/177_0069.jpg',
        '144.minotaur/144_0070.jpg',
        '252.car-side-101/252_0070.jpg',
        '252.car-side-101/252_0068.jpg',
        '017.bowling-ball/017_0070.jpg',
        '252.car-side-101/252_0067.jpg',
        '234.tweezer/234_0070.jpg',
        '123.ketch-101/123_0066.jpg',
        '063.electric-guitar-101/063_0070.jpg',
        '101.head-phones/101_0069.jpg',
        '067.eyeglasses/067_0068.jpg',
        '184.sheet-music/184_0066.jpg',
        '052.crab-101/052_0068.jpg',
        '214.teepee/214_0067.jpg',
        '184.sheet-music/184_0067.jpg',
        '252.car-side-101/252_0066.jpg',
        '140.menorah-101/140_0074.jpg',
        '155.paperclip/155_0072.jpg',
        '201.starfish-101/201_0071.jpg',
        '252.car-side-101/252_0073.jpg',
        '252.car-side-101/252_0074.jpg',
        '144.minotaur/144_0073.jpg',
        '146.mountain-bike/146_0075.jpg',
        '214.teepee/214_0074.jpg',
        '131.lightbulb/131_0073.jpg',
        '184.sheet-music/184_0075.jpg',
        '137.mars/137_0075.jpg',
        '252.car-side-101/252_0072.jpg',
        '006.basketball-hoop/006_0075.jpg',
        '252.car-side-101/252_0071.jpg',
        '138.mattress/138_0072.jpg',
        '252.car-side-101/252_0075.jpg',
        '102.helicopter-101/102_0071.jpg',
        '184.sheet-music/184_0065.jpg',
        '111.house-fly/111_0065.jpg',
        '184.sheet-music/184_0063.jpg',
        '252.car-side-101/252_0062.jpg',
        '233.tuning-fork/233_0063.jpg',
        '172.revolver-101/172_0065.jpg',
        '084.giraffe/084_0063.jpg',
        '252.car-side-101/252_0061.jpg',
        '252.car-side-101/252_0065.jpg',
        '006.basketball-hoop/006_0063.jpg',
        '094.guitar-pick/094_0062.jpg',
        '252.car-side-101/252_0063.jpg',
        '252.car-side-101/252_0064.jpg',
        '232.t-shirt/232_0220.jpg',
        '232.t-shirt/232_0245.jpg',
        '232.t-shirt/232_0308.jpg',
        '234.tweezer/234_0104.jpg',
        '245.windmill/245_0088.jpg',
        '252.car-side-101/252_0081.jpg',
        '252.car-side-101/252_0082.jpg',
        '252.car-side-101/252_0083.jpg',
        '252.car-side-101/252_0084.jpg',
        '252.car-side-101/252_0085.jpg',
        '252.car-side-101/252_0086.jpg',
        '252.car-side-101/252_0087.jpg',
        '252.car-side-101/252_0088.jpg',
        '252.car-side-101/252_0089.jpg',
        '252.car-side-101/252_0090.jpg',
        '252.car-side-101/252_0091.jpg',
        '252.car-side-101/252_0092.jpg',
        '252.car-side-101/252_0093.jpg',
        '252.car-side-101/252_0094.jpg',
        '252.car-side-101/252_0095.jpg',
        '252.car-side-101/252_0096.jpg',
        '252.car-side-101/252_0097.jpg',
        '252.car-side-101/252_0098.jpg',
        '252.car-side-101/252_0099.jpg',
        '252.car-side-101/252_0100.jpg',
        '252.car-side-101/252_0101.jpg',
        '252.car-side-101/252_0102.jpg',
        '252.car-side-101/252_0103.jpg',
        '252.car-side-101/252_0104.jpg',
        '252.car-side-101/252_0105.jpg',
        '252.car-side-101/252_0106.jpg',
        '252.car-side-101/252_0107.jpg',
        '252.car-side-101/252_0108.jpg',
        '252.car-side-101/252_0109.jpg',
        '252.car-side-101/252_0110.jpg',
        '252.car-side-101/252_0111.jpg',
        '252.car-side-101/252_0112.jpg',
        '252.car-side-101/252_0113.jpg',
        '252.car-side-101/252_0114.jpg',
        '252.car-side-101/252_0115.jpg',
        '252.car-side-101/252_0116.jpg',
        '145.motorbikes-101/145_0511.jpg',
        '145.motorbikes-101/145_0512.jpg',
        '145.motorbikes-101/145_0561.jpg',
        '145.motorbikes-101/145_0570.jpg',
        '145.motorbikes-101/145_0576.jpg',
        '145.motorbikes-101/145_0640.jpg',
        '145.motorbikes-101/145_0659.jpg',
        '145.motorbikes-101/145_0679.jpg',
        '145.motorbikes-101/145_0754.jpg',
        '145.motorbikes-101/145_0771.jpg',
        '155.paperclip/155_0081.jpg',
        '159.people/159_0084.jpg',
        '159.people/159_0101.jpg',
        '159.people/159_0117.jpg',
        '159.people/159_0120.jpg',
        '159.people/159_0121.jpg',
        '159.people/159_0133.jpg',
        '159.people/159_0139.jpg',
        '159.people/159_0147.jpg',
        '159.people/159_0161.jpg',
        '159.people/159_0165.jpg',
        '159.people/159_0169.jpg',
        '159.people/159_0186.jpg',
        '159.people/159_0188.jpg',
        '159.people/159_0206.jpg',
        '172.revolver-101/172_0085.jpg',
        '172.revolver-101/172_0086.jpg',
        '173.rifle/173_0097.jpg',
        '177.saturn/177_0084.jpg',
        '184.sheet-music/184_0082.jpg',
        '184.sheet-music/184_0084.jpg',
        '205.superman/205_0084.jpg',
        '214.teepee/214_0129.jpg',
        '225.tower-pisa/225_0090.jpg',
        '229.tricycle/229_0095.jpg',
        '230.trilobite-101/230_0087.jpg',
        '230.trilobite-101/230_0090.jpg'
    ]
    return filename.split('256_ObjectCategories/')[-1] in blacklist


class ImageCoder(object):
    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes RGB JPEG data.
        self._raw_data = tf.placeholder(dtype=tf.string)
        self._image_data = tf.image.decode_image(self._raw_data)
        self._image_data = tf.image.grayscale_to_rgb(self._image_data)  # 1 channel to 3 channels
        self._encoded_data = tf.image.encode_jpeg(self._image_data, format='rgb', quality=100)

    def re_encode_jpeg(self, image_data):
        # since tf1.2, decode_jpeg can decode JPEGs, PNGs, BMPs and non-animated GIFs; so for compatibility,
        # re-encoding all of three to jpegs for version < 1.2.
        return self._sess.run(self._encoded_data,
                              feed_dict={self._raw_data: image_data})


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

    if _is2convert(filename):
        print 'Reencoding to JPEG for %s' % filename
        image_data = coder.re_encode_jpeg(image_data)

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
        output_filename = '%s-%.3d-of-%.3d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]

            image_buffer = _process_image(filename, coder)

            example = _convert_to_example(image_buffer, label, filename)
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
    filenames, labels = generate_lists(directory, name)
    _process_image_files(name, filenames, labels, num_shards)


def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.test_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with '
        'FLAGS.validation_shards')

    if os.path.exists(FLAGS.output_directory) is not True:
        os.mkdir(FLAGS.output_directory)

    # Run it!
    # _process_dataset('train', '/home/jacques/workspace/database/Caltech256/256_ObjectCategories', FLAGS.train_shards)
    # _process_dataset('test', '/home/jacques/workspace/database/Caltech256/256_ObjectCategories', FLAGS.test_shards)
    _process_dataset('rest', '/home/jacques/workspace/database/Caltech256/256_ObjectCategories', FLAGS.rest_shards)


if __name__ == '__main__':
    tf.app.run()
