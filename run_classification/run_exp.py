import sys
sys.path.append('../')

import datetime
import os

import tensorflow as tf
from database import dataset_reader
from model import resnet
from experiment_manager.utils import LogDir, sorted_str_dict

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resnet', type=str, default='resnet_v1_101', help='resnet_v1_50, resnet_v1_101, resnet_v1_152')
parser.add_argument('--server', type=int, default=0, help='local machine 0 or server 1 or 2')
parser.add_argument('--epsilon', type=float, default=0.00001, help='epsilon in bn layers')
parser.add_argument('--norm_only', type=int, default=0,
                    help='no beta nor gamma in fused_bn (1). Or with beta and gamma(0).')
parser.add_argument('--data_type', type=int, default=32, help='float32 or float16')
parser.add_argument('--database', type=str, default='dogs120', help='dogs120, caltech256, indoors67')
parser.add_argument('--color_switch', type=int, default=0, help='color switch or not')
parser.add_argument('--eval_only', type=int, default=0, help='only do the evaluation (1) or do train and eval (0).')
parser.add_argument('--resize_image', type=int, default=1, help='whether resizing images for training and testing.')

parser.add_argument('--separate_reg', type=int, default=0, help='separate regularizers for optimizer.')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--optimizer', type=str, default='mom', help='mom, sgd, more to be added')
parser.add_argument('--log_dir', type=str, default='0', help='according to gpu index and wd method')
parser.add_argument('--lrn_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--weight_decay_mode', type=int, default=1, help='weight decay mode')
parser.add_argument('--weight_decay_rate', type=float, default=0.01, help='weight decay rate for existing layers')
parser.add_argument('--weight_decay_rate2', type=float, default=0.01, help='weight decay rate for new layers')
parser.add_argument('--train_max_iter', type=int, default=9000, help='Maximum training iteration')
parser.add_argument('--snapshot', type=int, default=3000, help='snapshot every ')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for mom optimizer')
parser.add_argument('--fine_tune_filename', type=str, default='../z_pretrained_weights/resnet_v1_101.ckpt',
                    help='fine_tune_filename')
parser.add_argument('--fisher_filename', type=str, default='./fisher_exp.npy', help='filename of fisher matrix')
parser.add_argument('--resume_step', type=int, default=None, help='resume step')
parser.add_argument('--lr_policy', type=str, default='step', help='*step*, *poly*, *linear* to iteration')
parser.add_argument('--lr_step', type=str, default=None, help='list of lr rate decreasing step. Default None.')
parser.add_argument('--step_size', type=float, default=0.1,
                    help='Each lr_step, learning rate decreases . Default to 0.1')
parser.add_argument('--gpu_num', type=int, default=1, help='gpu num')
parser.add_argument('--ema_decay', type=float, default=0.9, help='ema decay of moving average in bn layers')
parser.add_argument('--blur', type=int, default=1, help='random blur: brightness/saturation/constrast')
parser.add_argument('--initializer', type=str, default='xavier', help='he or xavier')
parser.add_argument('--fix_blocks', type=int, default=0,
                    help='number of blocks whose weights will be fixed when training')
parser.add_argument('--save_first_iteration', type=int, default=0, help='whether saving the initial model')
parser.add_argument('--fisher_epsilon', type=float, default=0, help='clip value for fisher regularization')
parser.add_argument('--examples_per_class', type=int, default=60, help='examples per class')

parser.add_argument('--test_max_iter', type=int, default=None, help='maximum test iteration')
parser.add_argument('--test_with_multicrops', type=int, default=0, help='whether using multiple crops for testing.')
parser.add_argument('--test_batch_size', type=int, default=100, help='batch size used for test or validation')
FLAGS = parser.parse_args()


def train(resume_step=None):
    global_step = tf.get_variable('global_step', [], dtype=tf.int64,
                                  initializer=tf.constant_initializer(0), trainable=False)
    print '================',
    if FLAGS.data_type == 16:
        print 'using tf.float16 ====================='
        data_type = tf.float16
    else:
        print 'using tf.float32 ====================='
        data_type = tf.float32

    wd_rate_ph = tf.placeholder(data_type, shape=())
    wd_rate2_ph = tf.placeholder(data_type, shape=())
    lrn_rate_ph = tf.placeholder(data_type, shape=())

    with tf.variable_scope(FLAGS.resnet):
        images, labels, num_classes = dataset_reader.build_input(FLAGS.batch_size, 'train',
                                                                 examples_per_class=FLAGS.examples_per_class,
                                                                 dataset=FLAGS.database,
                                                                 resize_image=FLAGS.resize_image,
                                                                 color_switch=FLAGS.color_switch,
                                                                 blur=FLAGS.blur)
        model = resnet.ResNet(num_classes, lrn_rate_ph, wd_rate_ph, wd_rate2_ph,
                              optimizer=FLAGS.optimizer,
                              mode='train', bn_epsilon=FLAGS.epsilon, resnet=FLAGS.resnet, norm_only=FLAGS.norm_only,
                              initializer=FLAGS.initializer,
                              fix_blocks=FLAGS.fix_blocks,
                              fine_tune_filename=FLAGS.fine_tune_filename,
                              bn_ema=FLAGS.ema_decay,
                              wd_mode=FLAGS.weight_decay_mode,
                              fisher_filename=FLAGS.fisher_filename,
                              gpu_num=FLAGS.gpu_num,
                              fisher_epsilon=FLAGS.fisher_epsilon,
                              float_type=data_type,
                              separate_regularization=FLAGS.separate_reg)
        model.inference(images)
        model.build_train_op(labels)

    names = []
    num_params = 0
    for v in tf.trainable_variables():
        # print v.name
        names.append(v.name)
        num = 1
        for i in v.get_shape().as_list():
            num *= i
        num_params += num
    print "Trainable parameters' num: %d" % num_params

    precisions = tf.nn.in_top_k(tf.cast(model.predictions, tf.float32), model.labels, 1)
    precision_op = tf.reduce_mean(tf.cast(precisions, tf.float32))
    # ========================= end of building model ================================

    step = 0
    saver = tf.train.Saver(max_to_keep=0)
    logdir = LogDir(FLAGS.database, FLAGS.log_dir, FLAGS.weight_decay_mode)
    logdir.print_all_info()
    if not os.path.exists(logdir.log_dir):
        print 'creating ', logdir.log_dir, '...'
        os.mkdir(logdir.log_dir)
    if not os.path.exists(logdir.database_dir):
        print 'creating ', logdir.database_dir, '...'
        os.mkdir(logdir.database_dir)
    if not os.path.exists(logdir.exp_dir):
        print 'creating ', logdir.exp_dir, '...'
        os.mkdir(logdir.exp_dir)
    if not os.path.exists(logdir.snapshot_dir):
        print 'creating ', logdir.snapshot_dir, '...'
        os.mkdir(logdir.snapshot_dir)

    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    gpu_options = tf.GPUOptions(allow_growth=False)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=config)
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    import_variables = tf.trainable_variables()
    if FLAGS.fix_blocks > 0:
        import_variables = tf.global_variables()

    if FLAGS.fine_tune_filename is not None and resume_step is None:
        fine_tune_variables = []
        for v in import_variables:
            if 'logits' in v.name or 'Momentum' in v.name:
                print 'not loading %s' % v.name
                continue
            fine_tune_variables.append(v)

        loader = tf.train.Saver(var_list=fine_tune_variables)
        loader.restore(sess, FLAGS.fine_tune_filename)
        print('Succesfully loaded fine-tune model from %s.' % FLAGS.fine_tune_filename)
    elif resume_step is not None:
        # ./snapshot/model.ckpt-3000
        i_ckpt = logdir.snapshot_dir + '/model.ckpt-%d' % resume_step
        saver.restore(sess, i_ckpt)

        step = resume_step
        print('Succesfully loaded model from %s at step=%s.' % (i_ckpt, resume_step))
    else:
        print 'Not import any model.'

    print '=========================== training process begins ================================='
    f_log = open(logdir.exp_dir + '/' + str(datetime.datetime.now()) + '.txt', 'w')
    f_log.write('step,loss,precision,wd\n')
    f_log.write(sorted_str_dict(FLAGS.__dict__) + '\n')

    average_loss = 0.0
    average_precision = 0.0
    show_period = 20
    snapshot = FLAGS.snapshot
    max_iter = FLAGS.train_max_iter
    lrn_rate = FLAGS.lrn_rate

    lr_step = []
    if FLAGS.lr_step is not None:
        temps = FLAGS.lr_step.split(',')
        for t in temps:
            lr_step.append(int(t))

    t0 = None
    wd_rate = FLAGS.weight_decay_rate
    wd_rate2 = FLAGS.weight_decay_rate2
    while step < max_iter + 1:
        step += 1

        if FLAGS.lr_policy == 'step':
            if len(lr_step) > 0 and step == lr_step[0]:
                lrn_rate *= FLAGS.step_size
                lr_step.remove(step)
        elif FLAGS.lr_policy == 'poly':
            lrn_rate = ((1 - 1.0 * (step-1) / max_iter) ** 0.9) * FLAGS.lrn_rate
        elif FLAGS.lr_policy == 'linear':
            lrn_rate = FLAGS.lrn_rate / step
        else:
            lrn_rate = FLAGS.lrn_rate

        _, loss, wd, precision = sess.run(
            [model.train_op, model.loss, model.wd, precision_op],
            feed_dict={
                lrn_rate_ph: lrn_rate,
                wd_rate_ph: wd_rate,
                wd_rate2_ph: wd_rate2
            }
        )

        average_loss += loss
        average_precision += precision

        if FLAGS.save_first_iteration == 1 or step % snapshot == 0:
            saver.save(sess, logdir.snapshot_dir + '/model.ckpt', global_step=step)

        if step % show_period == 0:
            left_hours = 0

            if t0 is not None:
                delta_t = (datetime.datetime.now() - t0).seconds
                left_time = (max_iter - step) / show_period * delta_t
                left_hours = left_time/3600.0

            t0 = datetime.datetime.now()

            average_loss /= show_period
            average_precision /= show_period

            if step == 0:
                average_loss *= show_period
                average_precision *= show_period

            f_log.write('%d,%f,%f,%f\n' % (step, average_loss, average_precision, wd))
            f_log.flush()

            print '%s %s] Step %s, lr = %f, wd_rate = %f, wd_rate_2 = %f ' \
                  % (str(datetime.datetime.now()), str(os.getpid()), step, lrn_rate, wd_rate, wd_rate2)
            print '\t loss = %.4f, precision = %.4f, wd = %.4f' % (average_loss, average_precision, wd)
            print '\t estimated time left: %.1f hours. %d/%d' % (left_hours, step, max_iter)

            average_loss = 0.0
            average_precision = 0.0

    coord.request_stop()
    coord.join(threads)

    return f_log, logdir  # f_log returned for eval.


def eval(i_ckpt):
    tf.reset_default_graph()

    print '================',
    if FLAGS.data_type == 16:
        print 'using tf.float16 ====================='
        data_type = tf.float16
    else:
        print 'using tf.float32 ====================='
        data_type = tf.float32

    with tf.variable_scope(FLAGS.resnet):
        images, labels, num_classes = dataset_reader.build_input(FLAGS.test_batch_size,
                                                                 'val',
                                                                 dataset=FLAGS.database,
                                                                 color_switch=FLAGS.color_switch,
                                                                 blur=0,
                                                                 resize_image=FLAGS.resize_image,
                                                                 multicrops_for_eval=FLAGS.test_with_multicrops)
        model = resnet.ResNet(num_classes, None, None, None,
                              mode='eval', bn_epsilon=FLAGS.epsilon, norm_only=FLAGS.norm_only, resnet=FLAGS.resnet,
                              float_type=data_type)
        logits = model.inference(images)
        model.compute_loss(labels, logits)

    precisions = tf.nn.in_top_k(tf.cast(model.predictions, tf.float32), model.labels, 1)
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

    if i_ckpt is not None:
        loader = tf.train.Saver(max_to_keep=0)
        loader.restore(sess, i_ckpt)
        eval_step = i_ckpt.split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' % (i_ckpt, eval_step))

    print '======================= eval process begins ========================='
    average_loss = 0.0
    average_precision = 0.0
    if FLAGS.test_max_iter is None:
        max_iter = dataset_reader.num_per_epoche('eval', FLAGS.database) / FLAGS.test_batch_size
    else:
        max_iter = FLAGS.test_max_iter

    if FLAGS.test_with_multicrops == 1:
        max_iter = dataset_reader.num_per_epoche('eval', FLAGS.database)

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

    coord.request_stop()
    coord.join(threads)

    return average_loss / max_iter, average_precision / max_iter


def main(_):
    # ============================================================================
    # ============================= TRAIN ========================================
    # ============================================================================
    print sorted_str_dict(FLAGS.__dict__)
    if FLAGS.resume_step is not None:
        print 'Ready to resume from step %d.' % FLAGS.resume_step

    assert FLAGS.gpu_num is not None, 'should specify the number of gpu.'
    assert FLAGS.gpu_num > 0, 'the number of gpu should be bigger than 0.'
    if FLAGS.eval_only:
        logdir = LogDir(FLAGS.database, FLAGS.log_dir, FLAGS.weight_decay_mode)
        logdir.print_all_info()
        f_log = open(logdir.exp_dir + '/' + str(datetime.datetime.now()) + '.txt', 'w')
        f_log.write('step,loss,precision,wd\n')
        f_log.write(sorted_str_dict(FLAGS.__dict__) + '\n')
    else:
        f_log, logdir = train(FLAGS.resume_step)

    # ============================================================================
    # ============================= EVAL =========================================
    # ============================================================================
    logdir.print_all_info()

    f_log.write('TEST:step,loss,precision\n')

    import glob
    i_ckpts = sorted(glob.glob(logdir.snapshot_dir + '/model.ckpt-*.index'), key=os.path.getmtime)

    for i_ckpt in i_ckpts:
        i_ckpt = i_ckpt.split('.index')[0]
        loss, precision = eval(i_ckpt)
        step = i_ckpt.split('-')[-1]
        print '%s %s] Step %s Test' % (str(datetime.datetime.now()), str(os.getpid()), step)
        print '\t loss = %.4f, precision = %.4f' % (loss, precision)
        f_log.write('TEST:%s,%f,%f\n' % (step, loss, precision))
        f_log.flush()

    f_log.close()


if __name__ == '__main__':
    tf.app.run()
