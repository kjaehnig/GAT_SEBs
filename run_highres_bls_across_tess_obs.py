import lightkurve as lk
import astropy.table as astab
import pandas as pd
import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm
from optparse import OptionParser
# %pylab inline
# pylab.rcParams['figure.figsize'] = (16, 8)
import warnings
import astropy.table as astab
from astropy.io import fits
import helper_functions as hf
import pickle as pk
import os
from matplotlib import pyplot as plt


print(astropy.__version__)


DD = hf.load_system_specific_directory()


def run_highres_bls_across_tess_obs(index=0, ngrid=100000):

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

    tic = tic_systems_of_interest[index]

    res_dir = DD + "highres_bls/"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    
    img_dir = DD +'highres_bls_plots/'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    infile = open(f"{DD}joker_TESS_lightcurve_files/TIC_{tic}_lightcurve_data.pickle",'rb')
    res = pk.load(infile)
    infile.close()

    jk_row = res['joker_param']

    
    dur_grid = np.exp(np.linspace(np.log(0.001),np.log(0.1),10))
    
#     npts = 5000
#     pmin = period_grid.min()
#     pmax = period_grid.max()
#     mindur = dur_grid.min()

    all_lk = res['lk_coll'].stitch(corrector_func=lambda x: x.remove_nans().normalize())
    
        
#     print(f"Running BLS.")

#     maxtime = all_lk.time.max().value
#     mintime = all_lk.time.min().value

#     freq_f = int( ((pmin**-1 - pmax**-1) * (maxtime - mintime)**2) / (npts * mindur) ) 

    x = all_lk.time
    y = all_lk.flux
    yerr = all_lk.flux_err

    cusBLS = astropy.timeseries.BoxLeastSquares(x, y, yerr)

    max_period = 2.*jk_row['MAP_P']
    min_period = 1.5 * max(dur_grid)
    print(f"min per:{min_period}, max dur: {max(dur_grid)}")      #0.5 * jk_row['MAP_P']
    nf =   ngrid   #5 * 10**5
    baseline = max(all_lk.time.value) - min(all_lk.time.value)
    
    min_f = 1. / max_period
    max_f = 1. / min_period
    print(max_f, min_f, baseline, nf, min(dur_grid))
    freq_f = float( (max_f - min_f) * baseline**2. ) / ( (nf - 1) * min(dur_grid) )
    
    period_grid = cusBLS.autoperiod(dur_grid, 
                                    maximum_period = max_period,
                                    minimum_period = min_period,
                                    frequency_factor=freq_f)
    # assert min(period_grid.value) >= max(dur_grid)
    print(min(period_grid), max(dur_grid))
#     print(nf, len(period_grid))
    assert nf==len(period_grid)
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

    fig,ax = plt.subplots(figsize=(10,7))\
    ax.set_title(f"TIC {tic}")
    ax.set_xlabel("phase [days]")
    ax.set_ylabel("flux [ppt]")
    lc_folded_bls = ax.scatter(hf.fold(all_lk.time.value, cusBLSperiod.value, cusBLSt0.value), 
               all_lk.flux.value, marker='o',s=0.5,
               c=all_lk.time.value - all_lk.time.min().value,
               cmap='inferno')

    hbins, num = hf.folded_bin_histogram(lks=all_lk, 
                                        bls_period=cusBLSperiod.value, 
                                        bls_t0=cusBLSt0.value)
    ax.plot(hbins, num, color='white', lw=6,zorder=10)
    ax.plot(hbins, num, color='cyan', lw=3,zorder=11)
    plt.savefig(f"{img_dir}TIC_{tic}_highres_folded_phase_curve.png",dpi=150)
    plt.close(fig)

    outfile = open(f"{res_dir}TIC_{tic}_highres_bls_params.pickle",'wb')
    pk.dump(res, outfile)
    outfile.close()
#     print("Finished.")

    
result = OptionParser()
result.add_option('-i', dest='index', default=0, type='int', 
                help='tic ID number of target (defaults to 20215451)')
result.add_option("--ngrid", dest='ngrid', default=100000, type='int',
                help='number of pts in period grid (default: 1e5)')


if __name__ == "__main__":
    opt,arguments = result.parse_args()
    run_highres_bls_across_tess_obs(**opt.__dict__)
