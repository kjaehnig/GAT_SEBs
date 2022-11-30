# import lightkurve as lk
# import astropy.table as astab
# import pandas as pd
# import numpy as np
# import astropy
# import sys
# from astropy.coordinates import SkyCoord
# from astropy import units as u
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from tqdm import tqdm
# import warnings
# import astropy.table as astab
# from astropy.io import fits
from optparse import OptionParser
import os
import sys
what_machine_am_i_on = sys.platform

# import helper_functions as hf
# warnings.filterwarnings('ignore',
#     message="WARNING (theano.tensor.opt): Cannot construct a scalar test value from a test value with no size:"
# )
# import os
# import pickle as pk
# import pymc3 as pm
# import pymc3_ext as pmx
# import aesara_theano_fallback.tensor as tt
# from celerite2.theano import terms, GaussianProcess
# from pymc3.util import get_default_varnames, get_untransformed_name, is_transformed_name
# import theano
# import exoplanet as xo

# import arviz as az
# from corner import corner



def wrapper_function(index=0,
                    mf=2,
                    nt=1000,
                    nd=500,
                    nc=2,
                    sf=5,
                    ndata=5000,
                    ns=5,
                    norun=0,
                    center=0):


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
    def load_system_specific_directory():

        what_machine_am_i_on = sys.platform

        if what_machine_am_i_on == 'darwin':
            print("running on macOS")
            return "/Users/karljaehnig/CCA_work/GAT/"
        if what_machine_am_i_on == 'linux' or what_machine_am_i_on == 'linux2':
            print("running on linux")
            return "/mnt/home/kjaehnig/"

    DD = load_system_specific_directory()

    if what_machine_am_i_on != 'darwin':

        theano_root = DD + f"mcmc_chains/"
        print(f'theano_root_dir = {theano_root}')
        if not os.path.exists(theano_root):
            os.mkdir(theano_root)

        theano_path = theano_root + f"HMC_{tic_systems_of_interest[index]}_c{nc}_nt{nt}_nd{nd}/"
        
        if os.path.exists(theano_path):
            shutil.rmtree(theano_path)
        
        os.mkdir(theano_path)
        os.environ["THEANO_FLAGS"] = f"base_compiledir={theano_path}"



    from pymc3_ind_model_rusty import load_construct_run_pymc3_model


    load_construct_run_pymc3_model(TIC_TARGET=tic_systems_of_interest[index],
                                    mult_factor=mf,
                                    Ntune=nt, 
                                    Ndraw=nd, 
                                    chains=nc, 
                                    sparse_factor=sf, 
                                    ndata=ndata,
                                    nsig=ns,
                                    norun=norun,
                                    center=center)


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
result.add_option("--ndata", dest='ndata', default=5000, type='int',
                help='max N out-of-transit data points to thin light curve to (default: 5000)')
result.add_option("--ns", dest='ns', default=5, type='int',
                help='number of sigma to consider in constructing isochrones BinMod distributions (default: 5)')
result.add_option("--norun", dest='norun', default=0, type='int',
                help='if 1 then perform MAP steps, sigmaclip, 2nd MAP steps, w/ no MCMC')
result.add_option("--center", dest='center', default=0, type='int',
                help="which period/t0 to center data on (0=TESS, 1=APOGEE)")

if __name__ == "__main__":
    opt,arguments = result.parse_args()
    wrapper_function(**opt.__dict__)