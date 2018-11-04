# encoding: utf-8

# tensorflow
import tensorflow as tf
import math
from model_part import conv2d
from model_part import fc

def inference(images, reuse=False, trainable=True):
    coarse1_conv = conv2d('coarse1', images, [11, 11, 3, 96], [96], [1, 4, 4, 1], padding='VALID', reuse=reuse, trainable=trainable)
    coarse1 = tf.nn.max_pool(coarse1_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    coarse2_conv = conv2d('coarse2', coarse1, [5, 5, 96, 256], [256], [1, 1, 1, 1], padding='VALID', reuse=reuse, trainable=trainable)
    coarse2 = tf.nn.max_pool(coarse2_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    coarse3 = conv2d('coarse3', coarse2, [3, 3, 256, 384], [384], [1, 1, 1, 1], padding='VALID', reuse=reuse, trainable=trainable)
    coarse4 = conv2d('coarse4', coarse3, [3, 3, 384, 384], [384], [1, 1, 1, 1], padding='VALID', reuse=reuse, trainable=trainable)
    coarse5 = conv2d('coarse5', coarse4, [3, 3, 384, 256], [256], [1, 1, 1, 1], padding='VALID', reuse=reuse, trainable=trainable)
    coarse6 = fc('coarse6', coarse5, [6*10*256, 4096], [4096], reuse=reuse, trainable=trainable)
    coarse7 = fc('coarse7', coarse6, [4096, 4070], [4070], reuse=reuse, trainable=trainable)
    coarse7_output = tf.reshape(coarse7, [-1, 55, 74, 1])
    return coarse7_output


def inference_refine(images, coarse7_output, keep_conv, reuse=False, trainable=True):
    fine1_conv = conv2d('fine1', images, [9, 9, 3, 63], [63], [1, 2, 2, 1], padding='VALID', reuse=reuse, trainable=trainable)
    fine1 = tf.nn.max_pool(fine1_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='fine_pool1')
    fine1_dropout = tf.nn.dropout(fine1, keep_conv)
    fine2 = tf.concat([fine1_dropout, coarse7_output], 3)
    fine3 = conv2d('fine3', fine2, [5, 5, 64, 64], [64], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
    fine3_dropout = tf.nn.dropout(fine3, keep_conv)
    fine4 = conv2d('fine4', fine3_dropout, [5, 5, 64, 1], [1], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
    return fine4


def loss(logits, depths, invalid_depths):
    logits_flat = tf.reshape(logits, [-1, 55*74])
    depths_flat = tf.reshape(depths, [-1, 55*74])
    invalid_depths_flat = tf.reshape(invalid_depths, [-1, 55*74])

    predict = tf.multiply(logits_flat, invalid_depths_flat)
    target = tf.multiply(depths_flat, invalid_depths_flat)
    d = tf.subtract(predict, target)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean(sum_square_d / 55.0*74.0 - 0.5*sqare_sum_d / math.pow(55*74, 2))
    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op
