import sys
sys.path.append('../')
from database import dataset_reader
import datetime
import os

from model import resnet
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('test_batch_size', 10, 'batch size used for test or validation.')
tf.app.flags.DEFINE_integer('num_classes', 1000, 'num_classes')
tf.app.flags.DEFINE_integer('labels_offset', 0, 'num_classes')
tf.app.flags.DEFINE_string('mode', 'val', 'train or val.')
tf.app.flags.DEFINE_integer('server', 0, 'local machine 0 or server 1')
tf.app.flags.DEFINE_string('pre_trained_filename', '../z_pretrained_weights/resnet_v1_101.ckpt',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('finetuned_filename',
                           None,
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('log_dir', '0', 'batch, fused_bn, batch_renorm or layer')
tf.app.flags.DEFINE_float('epsilon', 0.00001, 'epsilon in bn layers.')
tf.app.flags.DEFINE_integer('norm_only', 0, 'no beta nor gamma in fused_bn (1). Or with beta and gamma(0).')
tf.app.flags.DEFINE_integer('color_switch', 0, 'rgb in database.')
tf.app.flags.DEFINE_integer('top_k', 1, 'top_k precision.')
tf.app.flags.DEFINE_integer('test_max_iter', None, 'Maximum test iteration.')
tf.app.flags.DEFINE_string('resnet', 'resnet_v1_101', 'resnet_v1_50, resnet_v1_101, resnet_v1_152')


def eval():
    with tf.variable_scope(FLAGS.resnet):
        images, labels, _ = dataset_reader.build_input(FLAGS.test_batch_size, 'val', dataset='imagenet',
                                                       blur=0,
                                                       color_switch=FLAGS.color_switch)
        model = resnet.ResNet(FLAGS.num_classes, None, None, None, resnet=FLAGS.resnet, mode=FLAGS.mode,
                              float_type=tf.float32)
        logits = model.inference(images)
        model.compute_loss(labels+FLAGS.labels_offset, logits)
        precisions = tf.nn.in_top_k(tf.cast(model.predictions, tf.float32), labels+FLAGS.labels_offset, FLAGS.top_k)

    precision_op = tf.reduce_mean(tf.cast(precisions, tf.float32))
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
                print 'last layer\'s variables: %s' % v.name
                continue

            print 'finetuned variables:', v.name
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

    print '======================= eval process begins ========================='
    average_loss = 0.0
    average_precision = 0.0
    if FLAGS.test_max_iter is None:
        max_iter = dataset_reader.num_per_epoche('eval', 'imagenet') / FLAGS.test_batch_size
    else:
        max_iter = FLAGS.test_max_iter

    step = 0
    while step < max_iter:
        step += 1
        loss, precision = sess.run([
            model.loss, precision_op
        ])

        average_loss += loss
        average_precision += precision
        if step % 100 == 0:
            print step, '/', max_iter, ':', average_loss / step, average_precision / step
        elif step % 10 == 0:
            print step, '/', max_iter, ':', average_loss / step, average_precision / step

        # batch size = 100, resnet_v1_101:
        # 10 / 500 : 1.05231621861 0.766999977827
        # 20 / 500 : 0.976500582695 0.773999977112
        # 30 / 500 : 0.969429596265 0.775666640202
        # 40 / 500 : 0.970155435801 0.773249974847
        # 50 / 500 : 0.980628492832 0.772999976873
        # 60 / 500 : 0.974383835991 0.771833313505
        # 70 / 500 : 0.967128694909 0.772428552594
        # 80 / 500 : 0.971453700215 0.768749982864
        # 90 / 500 : 0.977773231268 0.765444427066
        # 100 / 500 : 0.970571737289 0.76619998157
        # 460 / 500 : 0.990876104456 0.76347824374
        # 470 / 500 : 0.989712166723 0.763914876669
        # 480 / 500 : 0.989510885812 0.763645816346
        # 490 / 500 : 0.989354180499 0.764061207552
        # 500 / 500 : 0.992528787792 0.763559982777

        # batch size = 10, resnet_v1_101:
        # 10 / 5000 : 1.04412123561 0.769999998808
        # 20 / 5000 : 1.01021358222 0.784999996424
        # 30 / 5000 : 1.00341213942 0.779999997218
        # 40 / 5000 : 0.953749916703 0.77499999851
        # 50 / 5000 : 0.942187260389 0.775999999046
        # 100 / 5000 : 0.987873907052 0.764999999404
        # 5000 / 5000 : 0.992528784394 0.763559984207

        # batch size = 100, resnet_v1_152:
        # 10 / 500 : 0.894700920582 0.776999986172
        # 500 / 500 : 0.974335199773 0.7680999825

        # batch size = 100, resnet_v1_50:
        # 10 / 500 : 0.89778097868 0.774999976158
        # 500 / 500 : 1.04086481136 0.752079983711

        # https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

    coord.request_stop()
    coord.join(threads)

    return average_loss / max_iter, average_precision / max_iter


def main(_):
    loss, precision = eval()
    step = 0
    print '%s %s] Step %s Test' % (str(datetime.datetime.now()), str(os.getpid()), step)
    print '\t loss = %.4f, precision = %.4f' % (loss, precision)


if __name__ == '__main__':
    tf.app.run()
