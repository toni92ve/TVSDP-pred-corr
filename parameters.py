import copy, os, argparse
from pprint import pprint

"""
All user parameters are listed here, plus some handy function(s).
"""

_USER_PARAMETERS = { 

    "problem": { 
        # gamma1 is the step decrease factor. Another related parameter - step
        # increase factor - is obtained as a reciprocal: gamma2 = 1 / gamma1:
        "gamma1": 0.5,

        # gamma2
        "gamma2": 1.5,

        # Residual tolerance:
        "res_tol":  1e-4,

        # Penalization coefficient:
        "pen_coef":   1e-4,

        # Initial step size (delta t) in the inner loop:
        "ini_stepsize": .01,

        # Initial time
        "initial_time": 0.,

        # Final time
        "final_time": 1.,

        # Tolerance in constraints definition:
        "delta": 0.075,

        # Expansion factor of the delta parameter in case of QP solver failure:
        "delta_expand": 1.333,

        # Maximum number of delta expansions during one iteration:
        "max_delta_expansions": 5,

        # The smallest admissible delta t:
        "min_delta_t": 1e-10,

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

def getParameters(print_par: bool=True) -> dict:
    """
    Returns a deep copy of user parameters modified by those specified
    as command-line options.
    """
    # cmd_args = _ParseCommandLineArgs()

    assert isinstance(_USER_PARAMETERS, dict)
    user_params = copy.deepcopy(_USER_PARAMETERS)

    if print_par:
        print("-" * 80)
        print("User parameters:")
        pprint(user_params)
        print("-" * 80)
    return user_params

