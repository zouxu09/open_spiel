import reverb
import tensorflow as tf
import numpy as np

EPISODE_LENGTH = 100
OBSERVATION_SPEC = tf.TensorSpec([1530], tf.float32)
LEGALS_MASK_SPEC = tf.TensorSpec([2086], tf.uint8)
POLICY_SPEC = tf.TensorSpec([2086], tf.float32)
VALUE_SPEC = tf.TensorSpec([], tf.float32)

# Initialize the reverb server.
simple_server = reverb.Server(
    tables=[
        reverb.Table(
            name='my_table',
            sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
            remover=reverb.selectors.Fifo(),
            max_size=int(1e6),
            # Sets Rate Limiter to a low number for the examples.
            # Read the Rate Limiters section for usage info.
            rate_limiter=reverb.rate_limiters.MinSize(2),
            # The signature is optional but it is good practice to set it as it
            # enables data validation and easier dataset construction. Note that
            # we prefix all shapes with a 1 as the trajectories we'll be writing
            # consist of 1 timesteps.
            signature={
                'observation': tf.TensorSpec([1, *OBSERVATION_SPEC.shape], OBSERVATION_SPEC.dtype),
                'legals_mask': tf.TensorSpec([1, *LEGALS_MASK_SPEC.shape], LEGALS_MASK_SPEC.dtype),
                'policy': tf.TensorSpec([1, *POLICY_SPEC.shape], POLICY_SPEC.dtype),
                'value': tf.TensorSpec([1, *VALUE_SPEC.shape], VALUE_SPEC.dtype),
            },
        )
    ],
    # Sets the port to None to make the server pick one automatically.
    port=12345)

simple_server.wait()