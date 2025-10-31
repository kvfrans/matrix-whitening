###############################
#
#  Structures for managing training of flax networks.
#
###############################

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util
import optax
import functools
from typing import Any, Callable

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

# Contains model params and optimizer state.
class TrainStateEma(flax.struct.PyTreeNode):
    rng: Any
    step: int
    apply_fn: Callable = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    params_ema: Any
    ema_rate: float
    use_ema: bool
    tx: Any = nonpytree_field()
    opt_state: Any


    @classmethod
    def create(cls, model_def, params, rng, tx=None, opt_state=None, use_ema=False, **kwargs):
        if tx is not None and opt_state is None:
            opt_state = tx.init(params)
        params_ema = None if not use_ema else jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)

        return cls(
            rng=rng, step=1, apply_fn=model_def.apply, model_def=model_def, params=params, use_ema=use_ema,
            tx=tx, opt_state=opt_state, params_ema=params_ema, **kwargs,
        )

    # Call model_def.apply_fn.
    def __call__(self, *args, params=None, method=None, **kwargs,):
        if params is None:
            params = self.params
        variables = {"params": params}
        if isinstance(method, str):
            method = getattr(self.model_def, method)
        return self.apply_fn(variables, *args, method=method, **kwargs)

    def update_ema(self):
        new_params_ema = jax.tree_util.tree_map(
            lambda p, tp: p * (1-self.ema_rate) + tp * self.ema_rate, self.params, self.params_ema
        )
        return self.replace(params_ema=new_params_ema)

    @jax.jit
    def get_ema_params(self):
        return jax.tree_util.tree_map(lambda e: e / (1 - self.ema_rate ** self.step), self.params_ema)

    def call_model(self, *args, params=None, method=None, **kwargs):
        return self.__call__(*args, params=params, method=method, **kwargs)

    # For pickling.
    def save(self):
        return {
            'params': self.params,
            'params_ema': self.params_ema,
            'opt_state': self.opt_state,
            'step': self.step,
        }
    
    def load(self, data):
        return self.replace(**data)