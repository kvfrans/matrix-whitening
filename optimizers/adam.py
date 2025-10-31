import jax.experimental
import jax.experimental.multihost_utils
from optax import tree_utils as otu
import jax
from optimizers.optimizer_utils import Optimizer
import jax.numpy as jnp

def make_adam(
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        v_adafactor: bool = False,
    ):

    def init_fn(params):
        momentum = otu.tree_zeros_like(params)
        if not v_adafactor:
            variance = otu.tree_zeros_like(params)
        else:
            def v_decomp(p):
                if len(p.shape) == 2:
                    return [jnp.zeros((d,)) for d in p.shape]
                return jnp.zeros(p.shape)
            variance = jax.tree_util.tree_map(v_decomp, params)
        step = 0
        return (momentum, variance, step)
    
    def update_fn(grads, state, params):
        momentum, variance, step = state
        step = step + 1
        momentum = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, momentum, grads)
        if not v_adafactor:
            variance = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g ** 2, variance, grads)
        else:
            def v_update(g, v):
                if len(g.shape) == 2:
                    return [b2 * v[i] + (1 - b2) * jnp.mean(g ** 2, axis=1-i) for i in range(2)]
                else:
                    return b2 * v + (1 - b2) * g ** 2
            variance = jax.tree_util.tree_map(v_update, grads, variance)
        momentum_hat = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** step), momentum)
        if v_adafactor:
            def get_v_diag(v):
                if type(v) == list:
                    v_mat = jnp.outer(v[0], v[1])
                    v_mat = (jnp.shape(v[1])[0] * v_mat) / jnp.sum(v[0])
                    return v_mat
                else:
                    return v
            variance_form = jax.tree_util.tree_map(get_v_diag, variance, is_leaf=lambda v: type(v) == list)
        else:
            variance_form = variance
        variance_hat = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** step), variance_form)
        updates = jax.tree_util.tree_map(lambda m, v: m / (v ** 0.5 + eps), momentum_hat, variance_hat)
        return updates, (momentum, variance, step)
    
    def get_grad_params(state, params):
        return params

    @jax.jit
    def get_eval_params(state, params):
        return params
    
    return Optimizer(init_fn, update_fn, get_grad_params, get_eval_params, None, None)
