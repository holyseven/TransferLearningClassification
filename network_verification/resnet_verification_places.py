from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')
from database import dataset_reader
import datetime
import os

from model import resnet
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resnet', type=str, default='resnet_v1_152', help='resnet_v1_50, resnet_v1_101, resnet_v1_152')
parser.add_argument('--server', type=int, default=0, help='local machine 0 or server 1 or 2')
parser.add_argument('--epsilon', type=float, default=0.00001, help='epsilon in bn layers')
parser.add_argument('--norm_only', type=int, default=0,
                    help='no beta nor gamma in fused_bn (1). Or with beta and gamma(0).')
parser.add_argument('--log_dir', type=str, default='0', help='log dir')
parser.add_argument('--color_switch', type=int, default=1, help='color switch or not')
parser.add_argument('--labels_offset', type=int, default=0, help='num_classes')
parser.add_argument('--pre_trained_filename', type=str, default='../z_pretrained_weights/resnet_v1_152_places365.ckpt',
                    help='pre_trained_filename')
parser.add_argument('--finetuned_filename', type=str, default=None, help='finetuned_filename')
parser.add_argument('--test_max_iter', type=int, default=None, help='maximum test iteration')
parser.add_argument('--test_with_multicrops', type=int, default=0, help='whether using multiple crops for testing.')
parser.add_argument('--test_batch_size', type=int, default=100, help='batch size used for test or validation')
FLAGS = parser.parse_args()


def eval():
    with tf.variable_scope(FLAGS.resnet):
        images, labels, num_classes = dataset_reader.build_input(FLAGS.test_batch_size,
                                                                 'val',
                                                                 dataset='places365',
                                                                 color_switch=FLAGS.color_switch,
                                                                 blur=0,
                                                                 multicrops_for_eval=FLAGS.test_with_multicrops)
        model = resnet.ResNet(num_classes, None, None, None, resnet=FLAGS.resnet, mode='test',
                              float_type=tf.float32)
        logits = model.inference(images)
        model.compute_loss(labels+FLAGS.labels_offset, logits)

    precisions = tf.nn.in_top_k(tf.cast(model.predictions, tf.float32), labels+FLAGS.labels_offset, 1)
    precision_op = tf.reduce_mean(tf.cast(precisions, tf.float32))
    if FLAGS.test_with_multicrops == 1:
        precisions = tf.nn.in_top_k([tf.reduce_mean(model.predictions, axis=[0])], [labels[0]], 1)
        precision_op = tf.cast(precisions, tf.float32)
    # ========================= end of building model ================================

    gpu_options = tf.GPUOptions(allow_growth=False)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=config)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if FLAGS.pre_trained_filename is not None and FLAGS.finetuned_filename is not None:
        last_layer_variables = []
        finetuned_variables = []
        for v in tf.global_variables():
            if 'Momentum' in v.name:
                continue
            if v.name.find('logits') > 0:
                last_layer_variables.append(v)
                print('last layer\'s variables: %s' % v.name)
                continue

            print('finetuned variables:', v.name)
            finetuned_variables.append(v)

        loader1 = tf.train.Saver(var_list=finetuned_variables)
        loader1.restore(sess, FLAGS.finetuned_filename)

        loader2 = tf.train.Saver(var_list=last_layer_variables)
        loader2.restore(sess, FLAGS.pre_trained_filename)

        print('Succesfully loaded model from %s and %s.' % (FLAGS.finetuned_filename, FLAGS.pre_trained_filename))
    elif FLAGS.pre_trained_filename is not None:
        loader = tf.train.Saver()
        loader.restore(sess, FLAGS.pre_trained_filename)

        print('Succesfully loaded model from %s.' % FLAGS.pre_trained_filename)
    else:
        print('No models loaded...')

    print('======================= eval process begins =========================')
    average_loss = 0.0
    average_precision = 0.0
    if FLAGS.test_max_iter is None:
        max_iter = dataset_reader.num_per_epoche('eval', 'places365') // FLAGS.test_batch_size
    else:
        max_iter = FLAGS.test_max_iter

    if FLAGS.test_with_multicrops == 1:
        max_iter = dataset_reader.num_per_epoche('eval', 'places365')

    step = 0
    while step < max_iter:
        step += 1
        loss, precision = sess.run([
            model.loss, precision_op
        ])

        average_loss += loss
        average_precision += precision
        if step % 100 == 0:
            print(step, '/', max_iter, ':', average_loss / step, average_precision / step)
        elif step % 10 == 0:
            print(step, '/', max_iter, ':', average_loss / step, average_precision / step)

        # 10 / 365 : 1.82831767797 0.541999986768
        # 20 / 365 : 1.83651441932 0.543499980867
        # 30 / 365 : 1.82160102526 0.53766664962
        # 40 / 365 : 1.82607512772 0.540499981493
        # 50 / 365 : 1.81884565115 0.537599982023
        # 60 / 365 : 1.8105583032 0.54083331277
        # 70 / 365 : 1.81942391736 0.537714266351
        # 80 / 365 : 1.82481645197 0.53937498033
        # 90 / 365 : 1.8170008858 0.539888870385
        # 100 / 365 : 1.82069332361 0.538299981654
        # 110 / 365 : 1.81809715033 0.538909073309
        # 120 / 365 : 1.81752947768 0.539916648219
        # 130 / 365 : 1.81486196976 0.541692289939
        # 140 / 365 : 1.81867902449 0.540785696464
        # 150 / 365 : 1.81397258838 0.542666648626
        # 160 / 365 : 1.81897332892 0.541937481798
        # 170 / 365 : 1.82705717648 0.540705863868
        # 180 / 365 : 1.83086211814 0.538833316167
        # 190 / 365 : 1.83226556652 0.537999982426
        # 200 / 365 : 1.83346488535 0.537949982285
        # 210 / 365 : 1.82859564338 0.538428553229
        # 220 / 365 : 1.8310897605 0.537454527617
        # 230 / 365 : 1.83356597475 0.536869547289
        # 240 / 365 : 1.8296460549 0.537999982511
        # 250 / 365 : 1.82880885744 0.538159982085
        # 260 / 365 : 1.8355158627 0.537038444097
        # 270 / 365 : 1.83302205448 0.537518500951
        # 280 / 365 : 1.83109659127 0.537464267867
        # 290 / 365 : 1.83024783546 0.537206878539
        # 300 / 365 : 1.82938882232 0.537733315627
        # 310 / 365 : 1.8317064889 0.537225788351
        # 320 / 365 : 1.83259406239 0.537249981891
        # 330 / 365 : 1.83219493519 0.53715149688
        # 340 / 365 : 1.83274422358 0.536823511211
        # 350 / 365 : 1.83405183656 0.536828553166
        # 360 / 365 : 1.83265232974 0.536805537095
        # 2018-04-05 17:05:19.543344 18642] Step 0 Test
        #      loss = 1.8364, precision = 0.5362
        # 2018-04-05 17:42:57.255770 26263] Step 0 Test (Multi-crops)
        #      loss = 1.8639, precision = 0.5472

    coord.request_stop()
    coord.join(threads)

    return average_loss / max_iter, average_precision / max_iter


def main(_):
    loss, precision = eval()
    step = 0
    print('%s %s] Step %s Test' % (str(datetime.datetime.now()), str(os.getpid()), step))
    print('\t loss = %.4f, precision = %.4f' % (loss, precision))


if __name__ == '__main__':
    tf.app.run()
