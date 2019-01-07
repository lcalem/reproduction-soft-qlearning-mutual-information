# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from model import QNetwork
from utils.tf_utils import target_update, summary_op, clip_grads, hard_backup, soft_backup, check_pi, schedule, td_loss, \
    q_to_v
from utils.misc import LinearSchedule


class NeuralAgent(object):
    def __init__(self,
                 obs_shape,
                 act_shape,
                 config,
                 scope="neural_agent",
                 is_training=True):
        self._config = config
        self._obs_shape = obs_shape
        self._act_shape = act_shape
        self._scope = scope
        self._is_training = is_training
        self._t = 0  # count the number of timesteps with the environment
        self._summary_dict = {}
        self.set_seed(config.seed)

    def set_seed(self, seed):
        self._rng = np.random.RandomState(seed)
        tf.set_random_seed(seed)

    def _init_sess(self, ratio):
        config = tf.ConfigProto()
        if ratio and ratio < 1.0:
            print("reducing GPU ratio to %s" % ratio)
            config.gpu_options.per_process_gpu_memory_fraction = ratio

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), filename='model.ckpt')
        self.writer = tf.summary.FileWriter(logdir=self._config.restore_path + '/train')
        self.train_writer = tf.summary.FileWriter(logdir=self._config.restore_path + '/train')
        self.ep_summary = tf.summary.Summary()

    def _init_ph(self):
        with tf.name_scope("ph_op"):
            self.s = tf.placeholder(tf.float32, shape=(None,) + self._obs_shape, name='s')
            self.a = tf.placeholder(tf.uint8, shape=(None,), name='a')
            self.s1 = tf.placeholder(tf.float32, shape=(None,) + self._obs_shape, name='s1')
            self.r = tf.placeholder(tf.float32, shape=(None,), name='r')
            self.d = tf.placeholder(tf.float32, shape=(None,), name='d')

    def _init_graph(self):
        raise NotImplementedError()

    def _loss_op(self):
        raise NotImplementedError()

    def _train_op(self):
        with tf.name_scope("train_op"):
            opt = tf.train.RMSPropOptimizer(self._config.q_lr)
            gvs, norm = clip_grads(self.loss, params=self._q.vars, clip=self._config.grad_norm)
            self._train = opt.apply_gradients(gvs, global_step=self._global_step)
            self._summary_dict['norm'] = norm
            self._summary_dict["loss"] = self.loss

    def _copy_op(self):
        with tf.name_scope("copy_op"):
            self.copy = target_update(target=self._q_target.vars,
                                      source=self._q.vars)  # hard update of the target network

    def _summary_op(self):
        with tf.name_scope("summary_op"):
            self.summary = summary_op(t_dict=self._summary_dict)

    def _get_dict(self, batch):
        return {
            k: v for k, v in zip([self.s, self.a, self.r, self.s1, self.d], batch)
        }

    def build(self, gpu_ratio):
        self._global_step = tf.Variable(initial_value=1, dtype=tf.int32, name="global_step", trainable=False)
        self._init_ph()
        self._init_graph()
        self._loss_op()
        self._train_op()
        self._copy_op()
        self._summary_op()
        self._init_sess(ratio=gpu_ratio)
        tf.logging.info("Built graph with scope {}".format(self._scope))

    def act(self, state):
        pass

    def train(self, batch):
        feed_dict = self._get_dict(batch)
        loss, _ = self.sess.run([self.loss, self._train], feed_dict)
        return loss

    def summarize(self, batch=None, metrics=None):
        summary, gs = self.sess.run([self.summary, self._global_step], self._get_dict(batch))
        self.writer.add_summary(summary, global_step=gs)
        self.writer.flush()

        # if metrics is not None:
        #     # TODO this is shit...fix me
        #     for k, v in metrics.items():
        #         self.ep_summary.value.add(tag=k, simple_value=v)
        #     self.train_writer.add_summary(self.ep_summary, global_step=gs)
        #     self.train_writer.flush()

    def update(self):
        self.sess.run(self.copy)

    def save(self):
        try:
            self.saver.save(sess=self.sess, save_path=self._config.restore_path + '/model/',
                            global_step=self._global_step,
                            write_meta_graph=False)

            tf.logging.log(tf.logging.DEBUG,
                           "Model saved at global step {}".format(self.global_step))
        except (ValueError, TypeError, RuntimeError) as e:
            raise e

    def restore(self, path):
        try:
            ckpt = tf.train.latest_checkpoint(path)
            self.saver.restore(sess=self.sess, save_path=ckpt)
        except (tf.errors.NotFoundError, tf.errors.InvalidArgumentError) as e:
            raise e

    @property
    def global_step(self):
        return self.sess.run(self._global_step)

    @property
    def scope(self):
        return self._scope

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope)

    @property
    def scalar_summaries(self):
        return [k for k, v in self._summary_dict.items() if v.get_shape().ndims < 1]


class RegularizedAgent(NeuralAgent):
    def __init__(self, obs_shape, act_shape, config, scope="regularized_agent"):
        super(RegularizedAgent, self).__init__(obs_shape, act_shape, config, scope)

        self._beta = tf.get_variable(name='beta', shape=(),
                                     initializer=tf.constant_initializer(value=self._config.beta),
                                     trainable=False,
                                     dtype=tf.float32)

        self._rho = tf.get_variable(name='rho', shape=[1, self._act_shape],
                                    initializer=tf.constant_initializer(1. / self._act_shape),
                                    trainable=False,
                                    dtype=tf.float32)

        self._summary_dict["beta"] = self._beta

        self._beta_op = None

    def train(self, batch):
        feed_dict = self._get_dict(batch)
        loss, _, beta = self.sess.run([self.loss, self._train, self._beta_op], feed_dict)
        return loss

    @property
    def beta(self):
        return self.sess.run(self._beta)

    @property
    def rho(self):
        return self.sess.run(self._rho)

    @property
    def eps(self):
        return 0.


class SQL(RegularizedAgent):
    def __init__(self, obs_shape, act_shape, config):
        super(SQL, self).__init__(obs_shape, act_shape, config, scope='sql')

    def _init_graph(self):
        self._q = QNetwork(output_shape=self._act_shape,
                           config=self._config.model, scope=self.scope + '/local')

        self._q_target = QNetwork(output_shape=self._act_shape,
                                  config=self._config.model, scope=self.scope + '/target')

        self.q = self._q(self.s)
        self.q_target = self._q_target(self.s1)

        self.pi = check_pi(tf.nn.softmax(self._beta * self.q))

    def _loss_op(self):
        v = q_to_v(self.q, self.a, self._act_shape)
        v_soft = soft_backup(self.q_target, self._beta)

        # \hat Q in Equation 11:
        v_soft = self.r + (1. - self.d) * self._config.gamma * v_soft

        # Equation 11:
        self.loss = td_loss(v, v_soft)

        self._beta_op = schedule(self._beta, self._global_step, self._config.beta_lr, schedule="linear")

    def act(self, state):
        pi, q = self.sess.run([self.pi, self.q], {self.s: [state]})
        assert np.isclose(pi.sum(), 1.) and np.all(pi >= 0.)
        action = self._rng.choice(list(range(self._act_shape)), p=pi.flatten())
        return action, q.flatten()[action]

    @property
    def beta(self):
        return self.sess.run(self._beta)


class DQN(NeuralAgent):
    def __init__(self, obs_shape, act_shape, config, scope="dqn"):
        super(DQN, self).__init__(obs_shape, act_shape, config=config, scope=scope)
        self._eps = LinearSchedule(schedule_timesteps=int(self._config.expl_fraction * self._config.max_steps),
                                   final_p=self._config.eps, initial_p=1.)

        self._t = 0

    def _init_graph(self):

        self._q = QNetwork(output_shape=self._act_shape,
                           config=self._config.model, scope=self._scope + '/local')

        self._q_target = QNetwork(output_shape=self._act_shape,
                                  config=self._config.model, scope=self._scope + '/target')

        self.q = self._q(self.s)
        self.q_target = self._q_target(self.s1)

    def _loss_op(self):
        v = q_to_v(self.q, self.a, self._act_shape)
        v_target = hard_backup(self.q_target)
        v_target = self.r + (1. - self.d) * self._config.gamma * v_target
        self.loss = td_loss(v, v_target)

    def act(self, state):
        # TODO may put this in the graph, but then eps must be supplied as ph?
        q = self.sess.run(self.q, feed_dict={self.s: [state]}).flatten()
        if np.random.uniform() > self._eps.value:
            action = np.argmax(q)
        else:
            action = self._rng.randint(0, high=self._act_shape)
            self._eps.update(self._t)
        self._t += 1
        return action, q[action]

    @property
    def beta(self):
        return 0.

    @property
    def eps(self):
        return self._eps.value


class MIRL(RegularizedAgent):
    def __init__(self, obs_shape, act_shape, config):
        super(MIRL, self).__init__(obs_shape, act_shape, config=config, scope="mirl")

        # Add the epslion schedule here
        self._eps = LinearSchedule(schedule_timesteps=int(self._config.expl_fraction * self._config.max_steps),
                                   final_p=self._config.eps, initial_p=1.)

        # use this as internal counter for t of the schedule of the epsilon
        self._t = 0

    def _init_graph(self):
        # create a local and target estimators
        self._q = QNetwork(output_shape=self._act_shape, config=self._config.model,
                           scope=self.scope + '/local')
        self._q_target = QNetwork(output_shape=self._act_shape, config=self._config.model,
                                  scope=self.scope + '/target')

        self.q = self._q(self.s)
        self.q_target = self._q_target(self.s1)

        # define the beta variable maybe fixed at the moment # i called initial beta
        # define a prior variable here
        # define the policy as in sql with the prior equation 5
        z = tf.reduce_sum(self._rho * tf.exp(self._beta * self.q), axis=1)
        pi = self._rho * tf.exp(self._beta * self.q) / z
        self.pi = check_pi(pi)
        # define the prior update as a tensorflow op
        # to ensure it sums to 1 (rule out floating point instabilities)
        self._rho_op = schedule(self._rho, self.pi, self._config.rho_lr, schedule="ema")

    def _loss_op(self):
        v = q_to_v(self.q, self.a, self._act_shape)
        v_soft = soft_backup(self.q, self._beta, self._rho)

        # compute the soft update as in eq 12 b
        v_soft = self.r + (1.0 - self.d) * self._config.gamma * v_soft

        # compute the loss as in eq 12
        self.loss = td_loss(v, v_soft)

        # beta plays a fundamental role in the convergence of this method.
        # this scheduling it is too fast at least for this env
        # that's why we have to clip it. Linear scheduling is a meaningful option and we don't have to clip it
        # found this solution in another paper

        # # update beta line 8 of the MIRL algorithm
        self._beta_op = schedule(self._beta, self._global_step, self._config.beta_lr, schedule="linear")

        # # create the assign op
        # self._summary_dict.update(
        #     {"q": self.q, "q_target": self.q_target,
        #      # "pi": self.pi, 'rho': self.rho,
        #      'beta': self._beta,
        #      'loss': self.loss})

    def act(self, state):
        # get the policy and the prior from the graph

        pi, rho, q = self.sess.run([self.pi, self._rho, self.q], feed_dict={self.s: [state]})
        pi = pi.flatten()
        rho = rho.flatten()/rho.sum()

        assert np.isnan(pi).sum() == 0 and np.isnan(rho).sum() == 0

        # apply epsilon greedy as in dqn but using the uniform prior as in sectin "behavioral policy"
        if self._rng.uniform() > self._eps.value:
            action = np.argmax(pi)
        else:
            assert np.allclose(rho.sum(), 1.) and np.all(rho >= 0.), 'failing rho %s, sum is %s' % (rho, rho.sum())
            action = self._rng.choice(np.arange(self._act_shape), p=rho)
            self._eps.update(self._t)
        self._t += 1

        # update the rho
        self.sess.run(self._rho_op, feed_dict={self.s: [state]})

        return action, q.flatten()[action]

    def train(self, batch):
        feed_dict = self._get_dict(batch)
        loss, _, beta = self.sess.run([self.loss, self._train, self._beta_op], feed_dict)
        return loss

    @property
    def beta(self):
        return self.sess.run(self._beta)

    @property
    def eps(self):
        return self._eps.value


class GL(DQN, RegularizedAgent):
    def __init__(self, obs_shape, act_shape, config):
        super(GL, self).__init__(obs_shape, act_shape, config, scope='g_learning')

    def _init_graph(self):
        # in the paper the problem is stated as a cost min problem
        # is the same as this
        self._q = QNetwork(output_shape=self._act_shape,
                           config=self._config.model, scope=self._scope + '/local')

        self._q_target = QNetwork(output_shape=self._act_shape,
                                  config=self._config.model, scope=self._scope + '/target')

        self.q = self._q(self.s)  # this is g in the paper
        self.q_target = self._q_target(self.s1)

        self._beta_op = schedule(self._beta, tf.cast(self._global_step, dtype=tf.float32), self._config.beta_lr,
                                 schedule="linear")

        pi = self._rho * tf.exp(self._beta * self.q) / tf.reduce_sum(self._rho * tf.exp(self._beta * self.q),
                                                                     axis=1)
        self.pi = check_pi(pi)

    def _loss_op(self):
        # where are considering cost here  so c = -r

        g = q_to_v(self.q, self.a, act_shape=self._act_shape)
        v_soft = soft_backup(self.q_target, beta=self._beta, rho=self._rho)

        g_target = self.r + self._config.gamma * (1 - self.d) * v_soft  # (soft_v * (1-cond) + cond * hard_v)

        self.loss = td_loss(g, g_target)

    @property
    def eps(self):
        return self._eps.value

    @property
    def beta(self):
        return self.sess.run(self._beta)
