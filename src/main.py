from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from features import compute_features


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("mode", choices=["compute_features", "pca", "plot"])


parser.add_argument(
    "-p", "--path",
    action="store"
)

parser.add_argument(
    "-d", "--device", type=str, default=None, help="Device to use: cuda or cpu"
)

parser.add_argument(
    "-n", "--n_components", action="store", default=100
)

args = parser.parse_args()

if args.mode == "compute_features":
    compute_features(args)






