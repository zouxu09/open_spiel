import reverb
import tensorflow as tf

dataset = reverb.TrajectoryDataset.from_table_signature(
  server_address=f'localhost:12345',
  table='my_table',
  max_in_flight_samples_per_worker=10)

# Batches 2 sequences together.
batched_dataset = dataset.batch(2)

for sample in batched_dataset.take(1):
  print(sample)