# -*- coding: utf-8 -*-


import tensorflow as tf


class Model(object):
    """
    Base class for Tensorflow function approximator
    """

    def __init__(self, scope='base'):
        self._scope = scope

    def __str__(self):
        # scope, number of params, ouptut size
        pass

    def __call__(self, *args):
        raise NotImplementedError("Must be implemented")

    @property
    def scope(self):
        return self._scope

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope)


class SoftActor(Model):
    # base class for policy network
    def __init__(self, output_shape, config, scope='actor'):
        super(SoftActor, self).__init__(scope)
        self._output_shape = output_shape
        self._config = config

    def __call__(self, s, n_particles=1):
        h = tf.layers.flatten(s)
        h = tf.layers.dense(h, self._config['fc'], activation=tf.nn.relu)
        out = tf.layers.dense(h, self._output_shape)
        # latent = tf.random_normal(shape=(tf.shape(s)[0], self._output_shape))
        # latent = tf.random_normal(shape=(tf.shape(s)[0], n_particles, self._output_shape))
        # h = tf.concat([s, latent], axis=-1)
        # out = conv_block(s, self._config, output_shape=self._output_shape, scope=self._scope)
        # out = fc_block(h, self._config['fc'], output_shape=self._output_shape, scope=self._scope)
        # out = tf.squeeze(out, axis=-1)
        return tf.nn.softmax(out)


class QNetwork(Model):
    def __init__(self, output_shape, config, scope='q_network'):
        super(QNetwork, self).__init__(scope)
        self._output_shape = output_shape
        self._config = config

    def __call__(self, h):
        # h = tf.layers.flatten(h)
        # h = fc_block(h, self._config['fc'], output_shape=self._output_shape, scope=self._scope)
        h = conv_block(h, self._config, output_shape=self._output_shape, scope=self._scope)
        return h

        # TODO: check that it's actually equivalent
        # conv1 = tf.layers.conv2d(input, 32, 8, 4, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.relu)
        # conv2 = tf.layers.conv2d(conv1, 64, 4, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.relu)
        # conv3 = tf.layers.conv2d(conv2, 64, 3, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.relu)

        # flattened = tf.layers.flatten(conv3)
        # fc1 = tf.layers.fully_connected(flattened, 512)
        # self.predictions = tf.layers.fully_connected(fc1, self.output_nb)


def fc_block(x,
             layers,
             output_shape,
             act=tf.nn.relu,
             init=tf.contrib.layers.xavier_initializer,
             scope="fc_block",
             ):
    h = x
    with tf.variable_scope(scope):
        for n, units in enumerate(layers):
            h = tf.layers.dense(h, units=units, kernel_initializer=tf.initializers.variance_scaling(),
                                name="fc_{}".format(n))
            h = act(h)

        h = tf.layers.dense(h, units=output_shape, kernel_initializer=tf.initializers.variance_scaling(), name='out',
                            activation=None)

    return h


def conv_block(x, layers, output_shape, act=tf.nn.relu, init=tf.contrib.layers.xavier_initializer, scope='conv_block'):
    h = x
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for idx, (filters, ks, stride) in enumerate(layers['conv']):
            h = tf.layers.conv2d(h, filters, ks, stride, kernel_initializer=init(), padding="same",
                                 name="conv_{}".format(idx), activation=act)
        h = tf.layers.flatten(h)
        h = tf.layers.dense(h, units=layers['fc'], kernel_initializer=tf.initializers.variance_scaling, name="fc_0",
                            activation=act)
        h = tf.layers.dense(h, units=output_shape, kernel_initializer=tf.initializers.variance_scaling, name="out",
                            activation=None)
    return h
