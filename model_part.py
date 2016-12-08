import tensorflow as tf

TOWER_NAME = 'tower'
UPDATE_OPS_COLLECTION = '_update_ops_'


def _variable_with_weight_decay(name, shape, stddev, wd, trainable=True):
    var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_gpu(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def conv2d(scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse is True:
            scope.reuse_variables()
        kernel = _variable_with_weight_decay(
            'weights',
            shape=shape,
            stddev=0.01,
            wd=wd,
            trainable=trainable
        )
        conv = tf.nn.conv2d(inputs, kernel, stride, padding=padding)
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv_ = tf.nn.relu(bias, name=scope.name)
        return conv_


def fc(scope_name, inputs, shape, bias_shape, wd=0.04, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse is True:
            scope.reuse_variables()
        flat = tf.reshape(inputs, [-1, shape[0]])
        weights = _variable_with_weight_decay(
            'weights',
            shape,
            stddev=0.01,
            wd=wd,
            trainable=trainable
        )
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1))
        fc = tf.nn.relu_layer(flat, weights, biases, name=scope.name)
        return fc
