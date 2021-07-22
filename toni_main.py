import sys, traceback, os
import numpy as np

from predictor_corrector_toni import PredictorCorrector
from parameters_toni import getParameters

N = 3
rank = 2

predcorr = PredictorCorrector(N=N, rank=rank, params=user_params)

predcorr.run(D1=prevD, D2=D.batch(),
    M1=prevM, M2=M.batch(),
    Y0=Y, lam=lam, mu=mu)

    