import tensorflow as tf
from tensorflow.contrib.layers import layer_norm

from basic_model.model import Module


class Network(Module):
    """ Interface """
    def __init__(self,
                 name,
                 args,
                 graph,
                 obs,
                 env,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.obs = obs
        self.env = env
        self.action_dim = env.action_dim
        self.norm = layer_norm if 'layernorm' in args and args['layernorm'] else None

        super().__init__(name, 
                         args, 
                         graph, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    def _build_graph(self):
        self.action = self._policy(self.obs)

    def _policy(self, x):
        for i, u in enumerate(self.args['units']):
            x = self.dense_norm_activation(x, u, norm=None)
        
        x = self.dense(x, self.action_dim)

        return x