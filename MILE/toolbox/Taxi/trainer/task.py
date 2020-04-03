import argparse
import json
import os

from . import model

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--train_data_paths',
    help = 'GCS or local path to training data',
    required = True
    )
  parser.add_argument('--train_batch_size',
    help = 'Batch size for training steps',
    type = int,
    default = 512
    )
  parser.add_argument('--train_steps',
    help = 'Steps to run the training job for',
    type = int
    )
  parser.add_argument('--eval_steps',
    help = 'Number of steps to run evaluation for at each checkpoint',
    default = 10,
    type = int)
  parser.add_argument('--eval_data_paths',
    help = 'GCS or local path to evaluation data',
    required = True)
  parser.add_argument('--hidden_units',
    help = 'List of hidden layer sizes to use for DNN feature columns',
    nargs = '+',
    type = int,
    default = [125, 32, 4])
  parser.add_argument('--output_dir',
    help = 'GCS location to write checkpoints and export models',
    required = True)
  parser.add_argument('--job-dir',
    help = 'this model ignores this field, but it is required by GC',
    default = 'junk')
  parser.add_argument('--eval_delay_secs',
    help = 'How long to wait beafore running first evaluation',
    default = 10,
    type = int)
  parser.add_argument('--throttle_secs',
    help = 'Seconds between evaluations',
    default = 300,
    type = int)
  
  args = parser.parse_args()
  arguments = args.__dict__

  arguments.pop('job-dir', None)
  arguments.pop('job-dir', None)

  output_dir = arguments['output_dir']

  output_dir = os.path.join(
    output_dir,
    json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trail',''))

  # Run the training job
  model.train_and_evaluate(arguments)

