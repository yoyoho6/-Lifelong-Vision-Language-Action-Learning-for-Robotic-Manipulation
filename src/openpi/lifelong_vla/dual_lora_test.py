import flax.linen as nn
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.lifelong_vla import dual_lora
from openpi.lifelong_vla import gemma


def test_einsum_supports_nonleading_batch_axis_and_isolates_gradients():
    config = dual_lora.DualLoRAConfig(short_rank=2, long_rank=2, alpha=2.0, gate_context_dim=5)
    layer = dual_lora.Einsum(shape=(3, 2, 4, 3), config=config)
    x = jnp.ones((2, 7, 4))
    context = jnp.ones((2, 5))
    variables = layer.init(jax.random.key(0), "BSD,3KDH->3BSKH", x, context)

    output = layer.apply(variables, "BSD,3KDH->3BSKH", x, context)
    assert output.shape == (3, 2, 7, 2, 3)

    def loss(params):
        return jnp.sum(layer.apply({"params": params}, "BSD,3KDH->3BSKH", x, context, "short"))

    flat_grads = traverse_util.flatten_dict(jax.grad(loss)(variables["params"]), sep="/")
    short_norm = sum(float(jnp.linalg.norm(value)) for key, value in flat_grads.items() if "lora_short" in key)
    long_norm = sum(float(jnp.linalg.norm(value)) for key, value in flat_grads.items() if "lora_long" in key)
    assert short_norm > 0
    assert long_norm == 0


def test_tiny_dual_lora_gemma_forward():
    adapter = dual_lora.DualLoRAConfig(short_rank=2, long_rank=2, gate_context_dim=64)
    config = gemma.get_config("dummy", adapter)
    module = gemma.Module(configs=[config, config], embed_dtype="float32")
    embedded = [jnp.ones((2, 2, 64)), jnp.ones((2, 1, 64))]
    positions = jnp.tile(jnp.arange(3), (2, 1))
    mask = jnp.ones((2, 3, 3), dtype=bool)
    context = jnp.ones((2, 64))
    variables = nn.Module.init(
        module,
        jax.random.key(1),
        embedded,
        positions,
        mask,
        gate_context=context,
        gradient_mode="both",
        method=gemma.Module.__call__,
    )
    output, _ = module.apply(variables, embedded, positions, mask, gate_context=context, gradient_mode="both")
    assert [value.shape for value in output] == [(2, 2, 64), (2, 1, 64)]
    assert all(np.isfinite(np.asarray(value)).all() for value in output)
