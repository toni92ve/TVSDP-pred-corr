## Author(s): Albert Akhriev, albert_akhriev@ie.ibm.com
## IBM Research Ireland (c), 2020.

import copy, os, argparse
from pprint import pprint

"""
All user parameters are listed here, plus some handy function(s).
"""

_USER_PARAMETERS = {
    "io": {
        # Path to image sequence. "None" implies artificial image sequence:
        "path": None,

        # Template of image file name. Assume changedetection.net input data:
        "template": "in*.jpg",

        # If path is not specified, artificial image sequence will be generated
        # with the following parameters: (height, width, number_of_frames):
        "arti_image": (40, 60, 1000),

        # Output directory:
        "output_dir": "output",

        # File for writing video results. Useful to make a demonstration:
        "output_video_file": None,

        # Flag enables plotting in GUI window during processing:
        "gui_plot": False
    },

    "thresholder": {
        # Valid threshold: 0 < threshold < 1, otherwise threshold
        # will be estimated by the algorithm based on isoperimetric ratio:
        "preset_threshold": -1.0,

        # In case of artificial image sequence, we can do perfect thresholding,
        # which might be useful for debugging:
        "perfect_arti_image_thresholding": True,

        # Logarithmic transformation of diff. image before thresholding.
        # Experiments suggested usefulness of such a transformation:
        "log_transform": True,

        # Upper bound on isoperimetric ratio, (0 ... 0.5]:
        "max_isoperimetric_ratio": 0.5,

        # Number of points used for background projection by solving LP. Zero
        # or negative value means all points to be used (accurate but slow):
        "num_LP_points": int(500),

        # Verbosity level of LP solver:
        "LP_solver_verbosity": int(0)
    },

    "problem": {
        "ini_solution": "BM",
        # Balancing multiplier in the initial solution problem:
        "kappa": 0.25,

        # History size, i.e. the number of recent-most images in data matrix:
        "T": int(50),

        # Maximal rank of low-rank approximation X = Y * Y^T:
        "rank": int(8),

        # Tolerance for the residual in strongly active constraint definition:
        "eta1": 1e-2,

        # Minimal Lagrangian multiplier in strongly active constraint definition:
        "eta2": 1e-2,

        # gamma1 is the step decrease factor. Another related parameter - step
        # increase factor - is obtained as a reciprocal: gamma2 = 1 / gamma1:
        "gamma1": 0.7,

        # Residual tolerance:
        "res_tol": 1e-4,

        # Initial step size (delta tau) in the inner loop:
        "ini_stepsize": .1,

        # Initial step size (delta tau) in the inner loop:
        "final_time": 1.,

        # Tolerance in constraints definition:
        "delta": 0.075,

        # Expansion factor of the delta parameter in case of QP solver failure:
        "delta_expand": 1.333,

        # Maximum number of delta expansions during one iteration:
        "max_delta_expansions": 5,

        # The smallest admissible delta tau:
        "min_delta_tau": 0.01,

        # Max. number of retrying attempts if residual is large in
        # predictor-corrector algorithm:
        "max_retry_attempts": 5
    },

    # Verbosity level:
    "verbose": int(0),

    # Additional parameters for debugging or visualization:
    # mybg is 1 or 2
    "debug":  dict({"mygen": True})


}


def _ParseCommandLineArgs():
    """
    Parses and returns command-line parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--path2seq", type=str, default=None,
        help="path to folder that contains an image sequence (CDNet2014)")
    parser.add_argument("--outdir", type=str, default=None,
        help="name of output directory; otherwise use default path")
    parser.add_argument("--outvideo", type=str, default=None,
        help="name of output video-file; useful for demonstration")
    parser.add_argument("-p", "--plot", action="store_true",
        help="enable visualization")
    parser.add_argument("-v", "--verbose", type=int, default=0,
        help="verbosity level, <= 0 means silent mode")
    parser.add_argument("-o", "--onlyvideo", action="store_true",
        help="play video-sequence only; do not process the video-sequence")
    parser.add_argument("--baseline", action="store_true",
        help="use a baseline algorithm without low-rank approximation (test)")
    cmd_args = parser.parse_args()
    print("-" * 80)
    print("Command-line parameters:")
    pprint(cmd_args)
    print("-" * 80)
    return cmd_args


def getParameters(print_par: bool=True) -> dict:
    """
    Returns a deep copy of user parameters modified by those specified
    as command-line options.
    """
    cmd_args = _ParseCommandLineArgs()

    assert isinstance(_USER_PARAMETERS, dict)
    user_params = copy.deepcopy(_USER_PARAMETERS)
    if cmd_args.path2seq:
        assert isinstance(cmd_args.path2seq, str)
        user_params["io"]["path"] = cmd_args.path2seq

    if cmd_args.outdir:
        assert isinstance(cmd_args.outdir, str) and os.path.isdir(cmd_args.outdir)
        user_params["io"]["output_dir"] = cmd_args.outdir
    outdir = user_params["io"]["output_dir"]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    print("Output directory:", outdir)

    if cmd_args.outvideo:
        assert isinstance(cmd_args.outvideo, str)
        outvideo = os.path.join(outdir, os.path.basename(cmd_args.outvideo))
        user_params["io"]["output_video_file"] = outvideo
        print("Output video-file:", outvideo)

    assert isinstance(cmd_args.verbose, int) and cmd_args.verbose >= 0
    user_params["verbose"] = cmd_args.verbose

    user_params["debug"]["baseline_algo"] = bool(cmd_args.baseline)
    user_params["debug"]["onlyvideo"] = bool(cmd_args.onlyvideo)
    user_params["io"]["gui_plot"] = bool(cmd_args.plot)

    if print_par:
        print("-" * 80)
        print("User parameters:")
        pprint(user_params)
        print("-" * 80)
    return user_params

