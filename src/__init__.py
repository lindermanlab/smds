# Catch expected warnings from TFP
import src.warnings

# Default to float32 matrix multiplication on TPUs and GPUs
import jax
jax.config.update('jax_default_matmul_precision', 'highest')
