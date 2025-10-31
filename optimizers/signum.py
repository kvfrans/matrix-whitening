import jax.experimental
import jax.experimental.multihost_utils
from optax import tree_utils as otu
import jax
from optimizers.optimizer_utils import Optimizer
import jax.numpy as jnp

import wandb
import matplotlib.pyplot as plt

def make_signum(
        b1: float = 0.9,
        lookahead: float = 0,
    ):

    def init_fn(params):
        momentum = otu.tree_zeros_like(params)
        step = 0
        return (momentum, step)
    
    def update_fn(grads, state, params):
        momentum, step = state
        step = step + 1
        momentum = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, momentum, grads)
        m_lookahead = jax.tree_util.tree_map(lambda m, g: (1-lookahead) * m + lookahead * g, momentum, grads)
        updates = jax.tree_util.tree_map(lambda m: jnp.sign(m), m_lookahead)
        return updates, (momentum, step)
    
    def get_grad_params(state, params):
        return params

    def get_eval_params(state, params):
        return params
    
    return Optimizer(init_fn, update_fn, get_grad_params, get_eval_params, None, None)
