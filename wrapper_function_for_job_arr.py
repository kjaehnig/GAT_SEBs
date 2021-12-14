import lightkurve as lk
import astropy.table as astab
import pandas as pd
import numpy as np
import astropy
import sys
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import warnings
import astropy.table as astab
from astropy.io import fits
from optparse import OptionParser
import helper_functions as hf
warnings.filterwarnings('ignore',
    message="WARNING (theano.tensor.opt): Cannot construct a scalar test value from a test value with no size:"
)
import os
import pickle as pk
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
from pymc3.util import get_default_varnames, get_untransformed_name, is_transformed_name
import theano
import exoplanet as xo

import arviz as az
from corner import corner



def wrapper_function(index=0,
                    mf=2,
                    nt=1000,
                    nd=500,
                    nc=2,
                    sf=5,
                    ns=5):


    tic_systems_of_interest = [
        28159019,
        272074664,
        20215452,
        99254945,
        144441148,
        169820068,
        126232983,
        164458426,
        164527723,
        165453878,
        258108067,
        271548206,
        365204192
        ]


    from pymc3_ind_model_rusty import load_construct_run_pymc3_model


    load_construct_run_pymc3_model(TIC_TARGET=tic_systems_of_interest[index],
                                    mult_factor=mf,
                                    Ntune=nt, 
                                    Ndraw=nd, 
                                    chains=nc, 
                                    sparse_factor=sf, 
                                    nsig=ns)


result = OptionParser()

result.add_option('--index', dest='index', default=0, type='int', 
                help='indice of tic system array (defaults to 0)')
result.add_option("--mf", dest='mf', default=1, type='int',
                help='multiplicative factor by which to increase multivariate prior variances (default: 1)')
result.add_option("--nt", dest="nt", default=1000, type='int',
                help="number of tuning draws to perform during sampling (default: 1000)")
result.add_option("--nd", dest="nd", default=500, type='int',
                help="number of sample draws to perform during sampling (default: 500)")
result.add_option("--nc", dest='nc', default=2, type='int',
                help='number of chains to run during sampling (default: 2)')
result.add_option("--sf", dest='sf', default=5, type='int',
                help='how sparse to make the lightcurve data before running pymc3 (default: 5)')
result.add_option("--ns", dest='ns', default=5, type='int',
                help='number of sigma to consider in constructing isochrones BinMod distributions (default: 5)')


if __name__ == "__main__":
    opt,arguments = result.parse_args()
    wrapper_function(**opt.__dict__)