import numpy as np
import tensorflow as tf

from basic_model.model import Model
from algo.dagger.networks import Network
from env.gym_env import GymEnv
from replay.imitation_replay import ImitationReplay
from utility.losses import huber_loss
from utility.tf_utils import stats_summary


class Agent(Model):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 env_args, 
                 buffer_args, 
                 sess_config=None, 
                 save=False, 
                 log_tensorboard=False, 
                 log_params=False, 
                 log_stats=False, 
                 device=None):
        self.env = env = GymEnv(env_args)

        self.buffer = ImitationReplay(buffer_args['filename'], 
                                      int(float(buffer_args['capacity'])), 
                                      args['batch_size'],
                                      env.state_space,
                                      env.action_dim)

        super().__init__(name,
                         args,
                         sess_config=sess_config,
                         save=save,
                         log_tensorboard=log_tensorboard,
                         log_params=log_params,
                         log_stats=log_stats,
                         device=device)

    def act(self, state):
        state = state[None, :]
        action = self.sess.run(self.action, feed_dict={self.data['state']: state})

        return np.squeeze(action)

    def learn(self, i):
        if self.log_tensorboard:
            _, summary = self.sess.run([self.opt_op, self.graph_summary])
            if i % 100 == 0:
                self.writer.add_summary(summary, i)
                self.save()
        else:
            self.sess.run([self.opt_op])

    def _build_graph(self):
        self.data = self._prepare_data()
        self.net = Network('Network', self.args['network'], self.graph, self.data['state'], self.env, self.name)
        self.action = self.net.action
        self.loss = self._loss(self.action, self.data['action'])
        _, _, _, _, self.opt_op = self.net._optimization_op(self.loss)

        self._log_loss()

    def _prepare_data(self):
        with tf.name_scope('data'):
            sample_type = (tf.float32, tf.float32)
            sample_shape = (
                (None, *self.env.state_space),
                (None, self.env.action_dim)
            )
            samples =(tf.data.Dataset.from_generator(self.buffer, sample_type, sample_shape)
                      .prefetch(tf.data.experimental.AUTOTUNE)
                      .make_one_shot_iterator()
                      .get_next(name='samples'))

        state, action = samples

        data = {}
        data['state'] = state
        data['action'] = action

        return data

    def _loss(self, action, target_action):
        with tf.name_scope('loss'):
            loss = tf.losses.mean_squared_error(target_action, action)
    
        return loss

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.name_scope('loss_'):
                tf.summary.scalar('loss', self.loss)