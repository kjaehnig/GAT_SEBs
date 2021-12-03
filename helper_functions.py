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
import warnings
import astropy.table as astab
from astropy.io import fits
import os
warnings.filterwarnings('ignore',
    message="WARNING (theano.tensor.opt): Cannot construct a scalar test value from a test value with no size:"
)

import pickle as pk
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
import exoplanet as xo

import arviz as az
from corner import corner

from scipy.signal import savgol_filter
import wquantiles

dd = "/Users/kjaehnig/CCA_work/GAT/"

def docs_setup():
    """Set some environment variables and ignore some warnings for the docs"""
    import logging
    import warnings


    # Remove when Theano is updated
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Remove when arviz is updated
    warnings.filterwarnings("ignore", category=UserWarning)

    logger = logging.getLogger("theano.gof.compilelock")
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger("theano.tensor.opt")
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger("exoplanet")
    logger.setLevel(logging.DEBUG)

docs_setup()






def get_multiple_ranges(lk_coll):
    from itertools import groupby
    from operator import itemgetter

    ranges =[]
    inds = []
    data = [ii.sector for ii in lk_coll]
    for k,g in groupby(enumerate(data),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        inds.append([ii in group for ii in data])
        ranges.append((group[0],group[-1]))
        
    return data,ranges, inds


def get_texp_from_lightcurve(res):
    with fits.open(res['all_lks'].filename) as hdu:
        hdr = hdu[1].header

    texp = hdr["FRAMETIM"] * hdr["NUM_FRM"]
    texp /= 60.0 * 60.0 * 24.0
    print(texp, texp*60*60*24)

    return texp


def load_precompiled_pymc3_model_data(DD=None, TIC_TARGET=None, sparse_factor=5):
    import pickle as pk
    filename = f"{TIC_TARGET.replace(' ','_').replace('-','_')}_sf{int(sparse_factor)}_pymc3_data_dict"

    try:
        file = open(DD + "pymc3_data_dicts/"+filename, 'rb')
        data_dict = pk.load(file)
        file.close()
        return data_dict
    except:
        print(DD+"pymc3_data_dict/"+filename)
        raise Exception(f"There is no dict file for {TIC_TARGET} with SF= {int(sparse_factor)}")


def get_system_data_for_pymc3_model(TICID):
    allvis17 = astab.Table.read("/Users/kjaehnig/CCA_work/GAT/dr17_joker/allVisit-dr17-synspec.fits",hdu=1, format='fits')
    allstar17 = astab.Table.read("/Users/kjaehnig/CCA_work/GAT/dr17_joker/allStar-dr17-synspec-gaiaedr3-xm.fits")
    allstar17 = allstar17[(allstar17['bp_rp'] < 10) & (allstar17['phot_g_mean_mag'] < 25)]
    calibverr = astab.Table.read(dd+'dr17_joker/allVisit-dr17-synspec-min3-calibverr.fits', format='fits', hdu=1)

    file = open(f"/Users/kjaehnig/CCA_work/GAT/joker_TESS_lightcurve_files/{TICID.replace(' ','_').replace('-','_')}_highres_bls_params.pickle",'rb')
    blsres = pk.load(file)
    file.close()

    file = open(f"/Users/kjaehnig/CCA_work/GAT/joker_TESS_lightcurve_files/{TICID.replace(' ','_').replace('-','_')}_lightcurve_data.pickle","rb")
    res = pk.load(file)
    file.close()
    
        #     print(calibverr.info)
    # Grab cross-match IDs
    sysapodat = allvis17[allvis17['APOGEE_ID'] == res['joker_param']['APOGEE_ID']]

    ## joining calib RV_ERRs with the RVs
    sysapodat = astab.join(sysapodat, calibverr['VISIT_ID','CALIB_VERR'], keys=('VISIT_ID','VISIT_ID'))
    
    return (res, blsres, sysapodat)


def highres_secondary_transit_bls(res, blsres):
    
    jk_row = res['joker_param']

    
    dur_grid = np.exp(np.linspace(np.log(0.001),np.log(0.1),1000))
    
#     npts = 5000
#     pmin = period_grid.min()
#     pmax = period_grid.max()
#     mindur = dur_grid.min()

    all_lk = res['lk_coll'].stitch(corrector_func=lambda x: x.remove_nans().normalize().flatten())
    
    transit_mask = all_lk.create_transit_mask(
        period=blsres['period_at_max_power'].value,
        duration=5*blsres['duration_at_max_power'].value,
        transit_time=blsres['t0_at_max_power'].value
    )

    no_transit_lks = all_lk[~transit_mask]
        


    x = no_transit_lks.time
    y = no_transit_lks.flux
    yerr = no_transit_lks.flux_err

    cusBLS = astropy.timeseries.BoxLeastSquares(x, y, yerr)

    max_period = 100. * blsres['period_at_max_power'].value
    min_period = .011      #0.5 * jk_row['MAP_P']
    nf =   5 * 10**5    #5 * 10**5
    baseline = max(all_lk.time.value) - min(all_lk.time.value)
    
    min_f = 1. / max_period
    max_f = 1. / min_period
    
    freq_f = ( (max_f - min_f) * baseline**2. ) / ( (nf - 1) * min(dur_grid) )
    
    period_grid = np.array([blsres['period_at_max_power'].value])

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
    
    res['no_transit_lk'] = no_transit_lks
    

    
    return res


def estimate_ecosw(bls2res, blsres):
    
    bls1_t0 = blsres['t0_at_max_power']
    bls2_t0 = bls2res['t0_at_max_power']
    
    delT = (bls2_t0.value - bls1_t0.value) % blsres['period_at_max_power'].value
    
    ecosw_testval = ( (np.pi/4.) * ( (2.*delT / blsres['period_at_max_power'].value) - 1) )
    
    return ecosw_testval


def get_M1_R1_from_binary_model(TIC_TARGET, nsig=3, fig_dest=None):
    """
    FOR LATER CONSIDERATION:
    IMPLEMENT A RECURSIVE MAD-SIGMA CLIPPING THAT CONTINUES CLIPPING AT 3-MAD-SIGMA UNTIL THERE ARE 
    NO MORE CLIPPED DATA POINTS. THIS IS TO ALLOW THE ISOCHRONES POSTERIORS TO HAVE MULTIMODAL DISTRIBUTIONS
    THAT WONT AFFECT THE CONSTRUCTION OF THE MULTIVARIATE PRIOR.
    """
    from isochrones import BinaryStarModel

    ID = TIC_TARGET.split(' ')[1]
    mod = BinaryStarModel.load_hdf(f"/Users/kjaehnig/CCA_work/GAT/pymultinest_fits/tic_{ID}_binary_model_obj.hdf")
    m0,m1,r0,r1,mbol0,mbol1 = mod.derived_samples[['mass_0','mass_1','radius_0', 'radius_1','Mbol_0','Mbol_1']].values.T
    
    num, denom = np.argmin([np.median(m0), np.median(m1)]), np.argmax([np.median(m0), np.median(m1)])
    ms, mp = [m0,m1][num], [m0,m1][denom]

    rs, rp = [r0,r1][num], [r0,r1][denom]
    
    mbols,mbolp = [mbol0,mbol1][num], [mbol0,mbol1][denom]
    
    mp_madSD = np.median(abs(mp - np.median(mp))) * 1.4826
    rp_madSD = np.median(abs(rp - np.median(rp))) * 1.4826
    mbp_madSD = np.median(abs(mbolp - np.median(mbolp))) * 1.4826
    
    ms_madSD = np.median(abs(ms - np.median(ms))) * 1.4826
    rs_madSD = np.median(abs(rs - np.median(rs))) * 1.4826
    mbs_madSD = np.median(abs(mbols - np.median(mbols))) * 1.4826
    
    nsig = nsig
    mp_keep = abs(mp - np.median(mp)) < nsig * mp_madSD
    rp_keep = abs(rp - np.median(rp)) < nsig * rp_madSD
    mbp_keep = abs(mbolp - np.median(mbolp)) < nsig * mbp_madSD
    
    ms_keep = abs(ms - np.median(ms)) < nsig * ms_madSD
    rs_keep = abs(rs - np.median(rs)) < nsig * rs_madSD
    mbs_keep = abs(mbols - np.median(mbols)) < nsig * mbs_madSD
    
    
    cmplt_mask = (mp_keep) & (rp_keep) & (mbp_keep) & (ms_keep) & (rs_keep) & (mbs_keep)

    
    mod.derived_samples['logMp'] = np.log(mp)
    mod.derived_samples['logRp'] = np.log(rp)
    mod.derived_samples['logk'] = np.log(rs / rp)
    mod.derived_samples['logq'] = np.log(ms / mp)
    mod.derived_samples['logs'] = np.log(mbols / mbolp)
    
    fig = corner(az.from_dict(mod.derived_samples[['logMp','logRp','logk','logq','logs']].to_dict('list')))
    fig.axes[0].set_title(TIC_TARGET + f"\nN: {cmplt_mask.shape[0]}" +f"\nNmask: {cmplt_mask.sum()}" + f"\nNSigClip: {int(nsig)}" )


    corner(az.from_dict(mod.derived_samples[['logMp','logRp','logk','logq','logs']][cmplt_mask].to_dict('list')), 
           plot_contours=False, color='red', fig=fig, zorder=10)
    if fig_dest is None:
        plt.savefig(f"/Users/kjaehnig/CCA_work/GAT/figs/{TIC_TARGET}_isochrones_BinFitCorner_w_{int(nsig)}sigmaclip.png",dpi=150, bbox_inches='tight')
    else:
        plt.savefig(f"{fig_dest}/{TIC_TARGET}_isochrones_BinFitCorner_w_{int(nsig)}sigmaclip.png", dpi=150,bbox_inches='tight')
    
    m1 = mp[cmplt_mask]
    r1 = rp[cmplt_mask]
    log_k = np.log(rs / rp)[cmplt_mask]
    log_q = np.log(ms / mp)[cmplt_mask]
    log_s = np.log(mbols / mbolp)[cmplt_mask]

    
    M1R1_mvPrior = np.array([
                            np.log(m1),
                            np.log(r1),
                             log_q,
                            log_s
                          ]
                           )
    
    M1R1_mu = np.mean(M1R1_mvPrior, axis=-1)
    M1R1_cov = np.cov(M1R1_mvPrior)
    
    
    return (M1R1_mu, M1R1_cov, np.mean(log_k), np.std(log_k), np.mean(np.log(r1)), np.std(np.log(r1)), np.mean(log_s), np.std(log_s), M1R1_mvPrior)
            

def fold(x, period, t0):
    hp = 0.5 * period
    return (x - t0 + hp) % period - hp


def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def check_for_system_directory(TIC_TARGET, return_directories=False):
    import os
    
    tess_pwd = "/Users/kjaehnig/CCA_work/GAT/apotess_systems/"
    current_tess_files = os.listdir("/Users/kjaehnig/CCA_work/GAT/apotess_systems/")
    
    tic_dirname = TIC_TARGET.replace(" ",'_')
    tic_dirname = tic_dirname.replace("-",'_')
    
    tic_dir = f"{tic_dirname}_files"

    if tic_dir not in current_tess_files:
        print(f"No directory(s) found for {TIC_TARGET}.")
        print(f"Making directory(s) for {TIC_TARGET}.")
        
        os.mkdir(tess_pwd+tic_dir)
        
        main_dir = tess_pwd+tic_dir
        
        fig_dir = tess_pwd+tic_dir + "/figures"
        
        os.mkdir(fig_dir)
    else:
        print(f"There is already a directory for {TIC_TARGET}.")
    
    if return_directories:
        return (tess_pwd+tic_dir, 
                tess_pwd+tic_dir + "/figures/")
    

def check_for_system_directory_rusty_side(DD, TIC_TARGET, return_directories=False):

    tess_pwd = f"{DD}apotess_systems/".replace('//','/')
    current_tess_files = os.listdir(tess_pwd)

    tic_dirname = f"{TIC_TARGET.replace(' ','_').replace('-','_')}_files"

    if tic_dirname not in current_tess_files:
        print(f"No directory(s) found for {TIC_TARGET}.")
        print(f"Making directory(s) for {TIC_TARGET}.")

        os.mkdir(tess_pwd+tic_dirname)
        
        main_dir = tess_pwd+tic_dirname
        
        fig_dir = tess_pwd+tic_dirname + "/figures"
        
        os.mkdir(fig_dir)
    else:
        print(f"There is already a directory for {TIC_TARGET}.")
    
    if return_directories:
        return (tess_pwd+tic_dirname, 
                tess_pwd+tic_dirname + "/figures/")



def load_all_data_for_pymc3_model(TIC_TARGET, sparse_factor=1, nsig=3):
    # TIC_TARGET = 'TIC 20215452'

    res, blsres, sysapodat = get_system_data_for_pymc3_model(TIC_TARGET)

    sys_dest, fig_dest = check_for_system_directory(TIC_TARGET, return_directories=True)
    M1R1_mu, M1R1_cov, logk_mu, logk_sig, logr1_mu, logr1_sig, logs_mu, logs_sig, MVpriorDat = get_M1_R1_from_binary_model(
        TIC_TARGET, nsig=nsig, fig_dest=fig_dest)
    
    rv_time = astropy.time.Time(sysapodat['JD'], format='jd', scale='tcb')
    # print(sysapodat['MJD'])
    texp = get_texp_from_lightcurve(res)

    x_rv = rv_time.btjd
    y_rv = sysapodat['VHELIO'] - res['joker_param']['MAP_v0']
    yerr_rv = sysapodat['CALIB_VERR']


    model_lk_data = res['lk_coll'].stitch(corrector_func=lambda x: x.remove_nans().normalize())
    x =    model_lk_data.remove_nans().time.btjd
    y =    model_lk_data.remove_nans().flux.value
    yerr = model_lk_data.remove_nans().flux_err.value

    x_lk_ref = min(x)

    x_rv = x_rv - x_lk_ref

    x = x - x_lk_ref 

    yerr = 1e3*(yerr / np.median(y))
    y = (y / np.median(y) - 1)

    y *= 1e3


    def run_with_sparse_data(x,y,yerr, use_sparse_data=False, sparse_factor=5):
        if use_sparse_data:
            np.random.seed(68594)
            m = np.random.rand(len(x)) < 1.0 / sparse_factor
            x = x[m]
            y = y[m]
            yerr = yerr[m]
        return x,y,yerr

    x,y,yerr = run_with_sparse_data(x,y,yerr,True, sparse_factor=sparse_factor)


    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    yerr = np.ascontiguousarray(yerr, dtype=np.float64)


    x_rv = np.ascontiguousarray(x_rv, dtype=np.float64)
    y_rv = np.ascontiguousarray(y_rv, dtype=np.float64)
    yerr_rv = np.ascontiguousarray(yerr_rv, dtype=np.float64)

    bls_period = blsres['period_at_max_power'].value
    print(blsres['t0_at_max_power'].btjd-x_lk_ref)
    bls_t0 = blsres['t0_at_max_power'].btjd - x_lk_ref
    print('lightcurve N datapoints: ',len(x),len(y),len(yerr), 'transit_epoch: ',bls_t0)


#     apo_period = jk_row['MAP_P'].value
#     apo_t0 = map_t0 = astropy.time.Time(res['joker_param']['MAP_t0_bmjd'], format='mjd', scale='tcb').btjd

    lit_period = bls_period  #bls_period      ### THESE ARE THE TWO VARIABLES USED
    lit_t0 = bls_t0   #bls_t0             ### IN THE PYMC3 MODEL BELOW


    transit_mask = model_lk_data.create_transit_mask(
        period=blsres['period_at_max_power'].value,
        duration=5*blsres['duration_at_max_power'].value,
        transit_time=blsres['t0_at_max_power']
    )

    no_transit_lks = model_lk_data[~transit_mask]
    y_masked = 1000 * (no_transit_lks.flux.value / np.median(no_transit_lks.flux.value) - 1)
    lk_sigma = np.std(y_masked)
    print(lk_sigma)

    Ntrans = np.floor((x.max() - lit_t0) / lit_period)
    print(Ntrans)
    lit_tn = lit_t0  + Ntrans * lit_period

    bls2res = highres_secondary_transit_bls(res,blsres)
    ecosw_tv = estimate_ecosw(bls2res, blsres)

    return {
        'texp' : texp,
        'x' : x,
        'y' : y,
        'yerr' : yerr,
        'x_rv' : x_rv,
        'y_rv' : y_rv,
        'lk_sigma' : lk_sigma,
        'yerr_rv' : yerr_rv,
        'lit_period' : lit_period,
        'lit_t0' : lit_t0,
        'Ntrans' : Ntrans,
        'lit_tn' : lit_tn,
        'ecosw_tv' : ecosw_tv,
        'isores' : {
            'logM1Q' : (M1R1_mu, M1R1_cov),
            'logR1' : (logr1_mu, logr1_sig),
            'logk' : (logk_mu, logk_sig),
            'logs' : (logs_mu, logs_sig),
            'MVdat': MVpriorDat
        }
    }


def plot_MAP_rv_curve_diagnostic_plot(model, soln, extras, mask, 
                                        title='',
                                        DO_NOT_PLOT=True,
                                         RETURN_FILENAME=False,
                                      filename='',
                                     pymc3_model_dict=None):
    if mask is None:
        mask = np.ones(len(extras['x']), dtype=bool)
    
    if 'period' in soln.keys():
        period = soln['period']
    else:
        if pymc3_model_dict is not None:
            lit_tn = pymc3_model_dict['lit_tn']
            lit_t0 = pymc3_model_dict['lit_t0']
            Ntrans = pymc3_model_dict['Ntrans']

            period = (lit_tn - lit_t0) / Ntrans
    if pymc3_model_dict is not None:
        x_rv, y_rv, yerr_rv = pymc3_model_dict['x_rv'], pymc3_model_dict['y_rv'], pymc3_model_dict['yerr_rv']
        x, y, yerr = pymc3_model_dict['x'], pymc3_model_dict['y'], pymc3_model_dict['yerr']
    else:
        x_rv = extras['x_rv']
        y_rv = extras['y_rv']
        yerr_rv = extras['yerr_rv']
        x = extras['x']
        y = extras['y']
    t_lc_pred = np.linspace(x.min(), x.max(), 3000) 
    
    t0 = soln['t0']
    mean = soln['mean_rv']
    x_phase = np.linspace(-0.5*period, 0.5*period, 1000)
    

    
    with model:
        gp_pred = (
            pmx.eval_in_model(extras["gp_lc_pred"], soln) + soln["mean_lc"]
        )
        lc = (
            pmx.eval_in_model(extras["model_lc"](x), soln)
            - soln["mean_lc"]
        )

        y_rv_mod = pmx.eval_in_model(extras['model_rv'](x_phase + t0), soln) - soln['mean_rv']
    
    # fig, axes = plt.subplots(nrows=3)
    gsfig = GridSpec(nrows=140,ncols=100)
    fig = plt.figure(figsize=(12,8), constrained_layout=False)

    fig, ax4 = plt.subplots(figsize=(12,18))

    ax1 = fig.add_subplot(gsfig[:30,:])
    ax2 = fig.add_subplot(gsfig[30:60,:])
    ax3 = fig.add_subplot(gsfig[70:100,:])
    ax4 = fig.add_subplot(gsfig[110:,:])
    
    ax1.tick_params(labelbottom=False,direction='in')
    ax2.tick_params(bottom=True,direction='in')

    ax1.plot(x, y, "k.", alpha=0.2)
    ax1.plot(x, gp_pred, color="C1", lw=1)
    ax1.plot(x,lc,'C2')

    ax1.set_xlim(x.min(),x.min()+np.ceil(period)*5)# x.max())
    plt1_ylim = ax1.set_ylim()
#     ax1.set_xlim(x.max()-10,x.max())# x.max())

#     ax2.plot(x, y - gp_pred, "k.", alpha=0.2)
#     ax2.plot(x, lc, color="C2", lw=1)
    ax2.plot(x, y-lc, color='C1', lw=2)
    
    ax2.set_ylim(plt1_ylim[0],plt1_ylim[1])
    ax2.set_xlim(x.min(),x.min()+np.ceil(period)*5)# x.max())
#     ax2.set_xlim(x.max()-10,x.max())# x.max())
    
    ax1.set_ylabel("raw flux [ppt]")
    ax2.set_ylabel("de-trended flux [ppt]")
    ax2.set_xlabel("time [TBJD]")



    x_fold = (
        (extras["x"] - t0) % period / period
    )
    inds = np.argsort(x_fold)

    ax3.plot(x_fold[inds], extras["y"][inds] - gp_pred[inds], "k.", alpha=0.2)
    ax3.plot(x_fold[inds] - 1, extras["y"][inds] - gp_pred[inds], "k.", alpha=0.2)
#     ax2.plot(
#         x_fold[mask][inds],
#         extras["y"][mask][inds] - gp_pred[mask][inds],
#         "k.",
#         alpha=0.2,
#         label="data!",
#     )
#     ax2.plot(x_fold[inds] - 1, extras["y"][inds] - gp_pred, "k.", alpha=0.2)

    yval = extras["y"][inds] - gp_pred
    bins = np.linspace(0, 1, 75)
    num, _ = np.histogram(x_fold[inds], bins, weights=yval)
    denom, _ = np.histogram(x_fold[inds], bins)
#     ax2.plot(0.5 * (bins[:-1] + bins[1:]) - 1, num / denom, ".w")

    args = dict(lw=1)

    x_fold = (x - t0) % period / period
    inds = np.argsort(x_fold)
    ax3.plot(x_fold[inds], lc[inds], "C2", **args)
    ax3.plot(x_fold[inds] - 1, lc[inds], "C2", **args)

    ax3.set_xlim(-1, 1)
    ax3.set_ylabel("de-trended flux [ppt]")
    ax3.set_xlabel("phase")
    
    x_rv_fold = fold(x_rv, period, t0)
    ax4.plot(x_phase, y_rv_mod, "C0")
    ax4.errorbar(x_rv_fold, y_rv-mean, yerr=np.sqrt(np.exp(2.*soln['log_sigma_rv']) + yerr_rv**2.),
                 fmt='.',c='black',ecolor='red', label='RV obs')
    ax4.set_title(title)
    
    y1,y2 = ax4.set_ylim()
    x1,x2 = ax4.set_xlim()
    ax4.vlines(0.25*(x2-x1) + x1,
               0.25*(y2-y1) + y1,
               0.25*(y2-y1) + y1 + np.exp(soln['log_sigma_rv'])
              )
    filename = filename + '.png'
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    
    if RETURN_FILENAME:
        return filename


