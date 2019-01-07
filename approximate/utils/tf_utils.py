#!/usr/bin/env python
# -*- coding: utf-8 -*-

# add summary op
# add copy and hard copy

import numpy as np
import tensorflow as tf

EPS = 1e-6


def soft_backup(q, beta, rho=None):
    if rho is not None:
        return tf.log(tf.reduce_sum(rho * tf.exp(beta * q), axis=1) + EPS) / beta
    else:
        return tf.log(tf.reduce_sum(tf.exp(beta * q), axis=1) + EPS) / beta


def hard_backup(q):
    return tf.reduce_max(q, axis=1)


def check_pi(pi):
    is_nan = tf.cast(tf.is_nan(pi), dtype=tf.float32)
    return tf.cond(tf.reduce_sum(is_nan) > 0., lambda: is_nan / tf.reduce_sum(is_nan, axis=1, keepdims=True),
                   lambda: pi, name="replace_pi")


def schedule(v, *args, schedule="linear", clip=None):
    if schedule == "linear":
        next_v = linear_schedule(v, *args)
    elif schedule == "ema":
        next_v = ema(v, *args)
    else:
        next_v = v

    if clip is not None:
        next_v = tf.clip_by_value(next_v, 1 / clip, clip)
    return tf.assign(v, next_v, name=schedule)


def linear_schedule(x, y, w):
    if not isinstance(w, tf.Tensor):
        w = tf.convert_to_tensor(w)
    if y.dtype.is_integer:
        y = tf.cast(y, dtype=tf.float32)
    return y * w


def ema(x, y, w):
    if not isinstance(w, tf.Tensor):
        w = tf.convert_to_tensor(w)
    return x * (1. - w) + y * w


def q_to_v(q, a, act_shape):
    return tf.reduce_sum(q * tf.one_hot(a, depth=act_shape), axis=1)


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])


def td_loss(q, q_hat):
    with tf.name_scope('td_loss'):
        q_hat = tf.stop_gradient(q_hat)
        return tf.multiply(.5, tf.reduce_mean((q - q_hat) ** 2), name='td_loss')


def clip_grads(loss, params, clip=20.):
    grads = tf.gradients(ys=loss, xs=params)
    clipped_grads, norm = tf.clip_by_global_norm(grads, clip)
    gvs = [(g, v) for (g, v) in zip(clipped_grads, params)]
    return gvs, norm


def set_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-3):
    """Gaussian kernel with dynamic bandwidth.

    The bandwidth is adjusted dynamically to match median_distance / log(Kx).
    See [2] for more information.

    Args:
        xs(`tf.Tensor`): A tensor of shape (N x Kx x D) containing N sets of Kx
            particles of dimension D. This is the first kernel argument.
        ys(`tf.Tensor`): A tensor of shape (N x Ky x D) containing N sets of Kx
            particles of dimension D. This is the second kernel argument.
        h_min(`float`): Minimum bandwidth.

    Returns:
        `dict`: Returned dictionary has two fields:
            'output': A `tf.Tensor` object of shape (N x Kx x Ky) representing
                the kernel matrix for inputs `xs` and `ys`.
            'gradient': A 'tf.Tensor` object of shape (N x Kx x Ky x D)
                representing the gradient of the kernel with respect to `xs`.

    Reference:
        [2] Qiang Liu,Dilin Wang, "Stein Variational Gradient Descent: A General
            Purpose Bayesian Inference Algorithm," Neural Information Processing
            Systems (NIPS), 2016.
    """
    Kx, D = xs.get_shape().as_list()[-2:]
    Ky, D2 = ys.get_shape().as_list()[-2:]
    assert D == D2

    leading_shape = tf.shape(xs)[:-2]

    # Compute the pairwise distances of left and right particles.
    diff = tf.expand_dims(xs, -2) - tf.expand_dims(ys, -3)
    # ... x Kx x Ky x D
    dist_sq = tf.reduce_sum(diff ** 2, axis=-1, keepdims=False)
    # ... x Kx x Ky

    # Get median.
    input_shape = tf.concat((leading_shape, [Kx * Ky]), axis=0)
    values, _ = tf.nn.top_k(
        input=tf.reshape(dist_sq, input_shape),
        k=(Kx * Ky // 2 + 1),  # This is exactly true only if Kx*Ky is odd.
        sorted=True)  # ... x floor(Ks*Kd/2)

    medians_sq = values[..., -1]  # ... (shape) (last element is the median)

    h = medians_sq / np.log(Kx)  # ... (shape)
    h = tf.maximum(h, h_min)
    h = tf.stop_gradient(h)  # Just in case.
    h_expanded_twice = tf.expand_dims(tf.expand_dims(h, -1), -1)
    # ... x 1 x 1

    kappa = tf.exp(-dist_sq / h_expanded_twice)  # ... x Kx x Ky

    # Construct the gradient
    h_expanded_thrice = tf.expand_dims(h_expanded_twice, -1)
    # ... x 1 x 1 x 1
    kappa_expanded = tf.expand_dims(kappa, -1)  # ... x Kx x Ky x 1

    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded
    # ... x Kx x Ky x D

    return kappa, kappa_grad


def summary_op(t_dict):
    ops = []
    for k, t in t_dict.items():
        name = t.name.replace(':', '_')
        if t.get_shape().ndims < 1:
            op = tf.summary.scalar(name=name, tensor=t)
        else:
            op = tf.summary.histogram(name=name, values=t)
        ops.append(op)
    return tf.summary.merge(ops)


def target_update(target, source, tau=1.):
    assert len(target) == len(source) and len(target) > 0
    ops = [tf.assign(t, (1. - tau) * t + tau * s) for t, s in zip(target, source)]
    return tf.group(*ops)
