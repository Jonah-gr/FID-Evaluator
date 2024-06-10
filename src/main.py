from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from features import compute_features
from pca import run_pca
from fid import get_fid_scores

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("mode", choices=["compute_features", "pca", "fid"])


parser.add_argument("-r", "--real", action="store")

parser.add_argument("-f", "--fake", action="store")

parser.add_argument("-d", "--device", type=str, default=None, help="Device to use: cuda or cpu")

parser.add_argument("--noise", default="0.0 0.1 0.2 0.3 0.4")

parser.add_argument("--noise_types", type=str, default="gauss", help="Type of noise to apply: gauss, blur, rectangles, swirl, salt&pepper")

parser.add_argument("-n", "--n_components", action="store", default="100")

args = parser.parse_args()

if args.mode == "compute_features":
    compute_features(args)

if args.mode == "pca":
    run_pca(args)

if args.mode == "fid":
    get_fid_scores()
