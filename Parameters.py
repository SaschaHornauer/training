"""Command line arguments parser configuration."""
import argparse  # default python library for command line argument parsing
import os

parser = argparse.ArgumentParser(  # pylint: disable=invalid-name
    description='Train DNNs on model car data.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu', default=0, type=int, help='Cuda GPU ID')
parser.add_argument('--no-gpu', dest='no_gpu', action='store_true')
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--display', dest='display', action='store_true')
parser.add_argument('--no-display', dest='display', action='store_false')
parser.set_defaults(display=True)

parser.add_argument('--verbose', default=True, type=bool,
                    help='Debugging mode')
parser.add_argument('--aruco', default=True, type=bool, help='Use Aruco data')
parser.add_argument('--data-path', default='/hostroot/home/dataset/' +
                    'bair_car_data', type=str)
parser.add_argument('--resume-path', default=None, type=str, help='Path to' +
                    ' resume file containing network state dictionary')
parser.add_argument('--bkup', default=None, type=str, help='Path to' +
                    ' resume file containing network state dictionary')
parser.add_argument('--save-path', default='save', type=str, help='Path to' +
                    ' folder to save net state dictionaries.')


# nargs='+' allows for multiple arguments and stores arguments in a list
parser.add_argument(
    '--ignore',
    default=(
        'reject_run',
        'left',
        'out1_in2',
        'play',
        'racing'),
    type=str,
    nargs='+',
    help='Skips these labels in data.')

parser.add_argument('--require-one', default=(), type=str, nargs='+',
                    help='Skips data without these labels in data.')
parser.add_argument('--use-states', default=(1, 3, 5, 6, 7), type=str,
                    nargs='+', help='Skips data outside of these states.')

parser.add_argument('--nframes', default=2, type=int,
                    help='# timesteps of camera input')
parser.add_argument('--nsteps', default=10, type=int,
                    help='# of steps of time to predict in the future')
parser.add_argument('--stride', default=3, type=int,
                    help="number of timesteps between network predictions")

parser.add_argument('--print-moments', default=1000, type=int,
                    help='# of moments between printing stats')
parser.add_argument('--save-moments', default=100000, type=int,
                    help='# of moments between printing stats')

ARGS = parser.parse_args()

# Check for $DISPLAY being blank
if 'DISPLAY' not in os.environ:
    ARGS.display = False
