import lightkurve as lk
import astropy.table as astab
import pandas as pd
import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import math
# %pylab inline
# pylab.rcParams['figure.figsize'] = (16, 8)
import warnings
import astropy.table as astab
from astropy.io import fits
import pickle as pk
import multiprocessing


def run_bls_across_entire_observations(obs_file):
    
    jk_row = obs_file['joker_param']

    
    dur_grid = np.exp(np.linspace(np.log(0.01),np.log(0.1),5))
    
#     npts = 5000
#     pmin = period_grid.min()
#     pmax = period_grid.max()
#     mindur = dur_grid.min()

    all_lk = res['all_lks']
    
        
    print(f"Running BLS.")

#     maxtime = all_lk.time.max().value
#     mintime = all_lk.time.min().value

#     freq_f = int( ((pmin**-1 - pmax**-1) * (maxtime - mintime)**2) / (npts * mindur) ) 

    x = all_lk.time
    y = all_lk.flux
    yerr = all_lk.flux_err

    cusBLS = astropy.timeseries.BoxLeastSquares(x, y, yerr)

    period_grid = cusBLS.autoperiod(dur_grid, 
                                    maximum_period = 2.*jk_row['MAP_P'], 
                                    frequency_factor=1.0,
                                   minimum_n_transit=1)
    res = cusBLS.power(period_grid, dur_grid)

    maxpow = np.argmax(res['power'])
    cusBLSperiod = res['period'][maxpow]
    cusBLSt0 = res['transit_time'][maxpow]
    cusBLSdur = res['duration'][maxpow]
    cusBLSdepth = res['depth'][maxpow]

    res['period_at_max_power'] = cusBLSperiod
    res['t0_at_max_power'] = cusBLSt0
    res['depth_at_max_power'] = cusBLSdepth
    res['duration_at_max_power'] = cusBLSdur
    res['max_power'] = maxpow
        
    print("Finished.")

    

    
    return res



def mp_bls_fitter(nprocs=1):
    from multiprocessing import Queue

    ddir = '/Users/kjaehnig/CCA_work/GAT/'
    file = open(ddir+"big_lightcurve_dict_shortcadence_tess_obs","rb")
    tess_dict = pk.load(file)
    file.close()


    TESS_IDS = tess_dict.keys()
    tess_obs_list = [tess_dict[ii] for ii in TESS_IDS]

    def worker(obs_list, out_q):
        outdict = {}
        for jj in range(len(obs_list)):
            outdict[TESS_IDS[jj]] =\
                run_bls_across_entire_observations(
                    obs_list[jj]
                    )
        out_q.put(outdict)

    out_q = Queue()

    chunksize = int(math.ceil(len(TESS_IDS) / float(nprocs)))
    print(chunksize)
    procs = []

    for ii in range(nprocs):
        p = multiprocessing.Process(
                target=worker,
                args=(tess_obs_list[chunksize * ii:chunksize* (ii+1)],
                out_q)
                )
        procs.append(p)
        p.start()

    resultdict = {}
    for ii in range(nprocs):
        resultdict.update(out_q.get())
    for p in procs:
        p.join()

    file = open(ddir+'big_BLS_res_dict','wb')
    pk.dump(resultdict, file)
    file.close()

    print("--- COMPLETED ---")

if __name__ == '__main__':
    mp_bls_fitter(nprocs=6)