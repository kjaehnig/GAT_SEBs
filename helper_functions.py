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
import sys
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


# def readin_config_file():
#     """ Place a config_file.txt within the same repository as this python
#     in order for this function to work and return the data directory
#     """
#     with open("/Users/karljaehnig/Repositories/GAT_SEBs/config_file.txt",'r') as f:
#         content = f.read()
#     paths = content.split("\n")
#     print(paths)
#     for path in paths:
#         # print(f"data directory is set to {path.split(' = ')[1].strip(")}")
#         return path.split(' = ')[1].strip('"')


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



# def readin_config_file():
#     with open("config_file.txt",'r') as f:
#         content = f.read()
#     paths = content.split("\n")
#     for path in paths:
#         # print(f'data directory set to {path.split(" = ")[1].strip("'")}')
#         return path.split(' = ')[1].strip('"')


def load_system_specific_directory():

    what_machine_am_i_on = sys.platform

    if what_machine_am_i_on == 'darwin':
        print("running on macOS")
        return "/Users/karljaehnig/CCA_work/GAT/"
    if what_machine_am_i_on == 'linux' or what_machine_am_i_on == 'linux2':
        print("running on linux")
        return "/mnt/home/kjaehnig/"


DD = load_system_specific_directory()


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
    allvis17 = astab.Table.read(DD+"dr17_joker/allVisit-dr17-synspec.fits",hdu=1, format='fits')
    allstar17 = astab.Table.read(DD+"dr17_joker/allStar-dr17-synspec-gaiaedr3-xm.fits")
    allstar17 = allstar17[(allstar17['bp_rp'] < 10) & (allstar17['phot_g_mean_mag'] < 25)]
    calibverr = astab.Table.read(DD+'dr17_joker/allVisit-dr17-synspec-min3-calibverr.fits', format='fits', hdu=1)


    # if sys.platform in ['linux', 'linux2']:
        # bls_dir = 'ceph/highres_bls/'
    # else:
    bls_dir = 'joker_TESS_lightcurve_files/'
    file = open(DD+f"{bls_dir}{TICID.replace(' ','_').replace('-','_')}_highres_bls_params.pickle",'rb')
    blsres = pk.load(file)
    file.close()

    file = open(DD+f"joker_TESS_lightcurve_files/{TICID.replace(' ','_').replace('-','_')}_lightcurve_data.pickle","rb")
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


def get_isochrones_binmod_res(TIC_TARGET, nsig=3, fig_dest=None):
    """
    FOR LATER CONSIDERATION:
    IMPLEMENT A RECURSIVE MAD-SIGMA CLIPPING THAT CONTINUES CLIPPING AT 3-MAD-SIGMA UNTIL THERE ARE 
    NO MORE CLIPPED DATA POINTS. THIS IS TO ALLOW THE ISOCHRONES POSTERIORS TO HAVE MULTIMODAL DISTRIBUTIONS
    THAT WONT AFFECT THE CONSTRUCTION OF THE MULTIVARIATE PRIOR.
    """
    from isochrones import BinaryStarModel

    if sys.platform in ['linux','linux2']:
        BinResDir = DD + 'ceph/pymultinest_fits/'
    else:
        BinResDir = DD + "pymultinest_fits_rusty/"
    ID = TIC_TARGET.split(' ')[1]
    mod = BinaryStarModel.load_hdf(BinResDir + f"tic_{ID}_binary_model_obj.hdf")
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
        plt.savefig(DD+f"figs/{TIC_TARGET}_isochrones_BinFitCorner_w_{int(nsig)}sigmaclip.png",dpi=150, bbox_inches='tight')
    else:
        plt.savefig(f"{fig_dest}/{TIC_TARGET}_isochrones_BinFitCorner_w_{int(nsig)}sigmaclip.png", dpi=150,bbox_inches='tight')
    
    m1 = mp[cmplt_mask]
    r1 = rp[cmplt_mask]
    log_k = np.log(rs / rp)[cmplt_mask]
    log_q = np.log(ms / mp)[cmplt_mask]
    log_s = np.log(mbols / mbolp)[cmplt_mask]

    
    mvPrior = np.array([
                            np.log(m1),
                            np.log(r1),
                             log_q,
                            log_s
                          ]
                           )
    
    mvPrior_mu = np.mean(mvPrior, axis=-1)
    mvPrior_cov = np.cov(mvPrior)
    
    
    return {'mvPrior_mu':mvPrior_mu, 
            'mvPrior_cov':mvPrior_cov,
            'logm1':[np.mean(np.log(m1)), np.std(np.log(m1))],
            'logr1':[np.mean(np.log(r1)), np.std(np.log(r1))],
            'logq':[np.mean(log_q), np.std(log_q)],
            'logs':[np.mean(log_s), np.std(log_s)], 
            'logk':[np.mean(log_k), np.std(log_k)], 
            'mvPrior':mvPrior
            }
          


def compute_value_in_post(model, idata, target, size=None):
    # Get the names of the untransformed variables
    vars = get_default_varnames(model.unobserved_RVs, True)
    names = list(sorted(set([
        get_untransformed_name(v.name)
        if is_transformed_name(v.name)
        else v.name
        for v in vars
    ])))

    # Compile a function to compute the target
    func = theano.function([model[n] for n in names], target, on_unused_input="ignore")

    # Call this function for a bunch of values
    flat_samps = idata.posterior.stack(sample=("chain", "draw"))
    if size is None:
        indices = np.arange(len(flat_samps.sample))
    else:
        indices = np.random.randint(len(flat_samps.sample), size=size)

    return [func(*(flat_samps[n].values[..., i] for n in names)) for i in indices]



def print_out_stellar_type(M,R):

    O_cond = (M >= 16) & (R >= 6.6)
    B_cond = (M >= 2.1) & (M < 16) & (R >= 1.8) & (R < 6.6)
    A_cond = (M >= 1.4) & (M < 2.1) & (R >= 1.4) & (R < 1.8)
    F_cond = (M >= 1.04) & (M < 1.4) & (R >= 1.15) & (R < 1.4)
    G_cond = (M >= 0.8) & (M < 1.04) & (R >= 0.96) & (R < 1.15)
    K_cond = (M >= 0.45) & (M < 0.8) & (R >= 0.7) & (R < 0.96)
    M_cond = (M >= 0.08) & (M < 0.45) & (R < 0.7)

    all_stellar_types = ['O','B','A','F','G','K','M']
    all_stellar_conds = [O_cond, B_cond, A_cond, F_cond,
                        G_cond, K_cond, M_cond]

    mask = [i for i, x in enumerate(all_stellar_conds) if x]
    if np.any(mask):
        return all_stellar_types[mask[0]]
    if np.all(mask):
        return 'None'


def write_a_story_for_system(TIC_TARGET='TIC 20215452', model_type='1x',
                    Ntune=1000, Ndraw=500, chains=4, return_dict=False):

    file = open(f"/Users/karljaehnig/CCA_work/GAT/pymc3_models/{TIC_TARGET}_pymc3_Nt{Ntune}_Nd{Ndraw}_Nc{chains}_individual_priors_{model_type}_isochrones.pickle",'rb')
    res_dict = pk.load(file)
    file.close()

    flat_samps = res_dict['trace'].posterior.stack(sample=('chain','draw'))

    m1 = flat_samps['M1'].median().values
    r1 = flat_samps['R1'].median().values
    logg1 = np.log(m1) - 2.*np.log(r1) + 4.437
    stype1 = print_out_stellar_type(m1,r1)

    m2 = flat_samps['M2'].median().values
    r2 = flat_samps['R2'].median().values
    logg2 = np.log(m2) - 2.*np.log(r2) + 4.437
    stype2 = print_out_stellar_type(m2,r2)

    a = flat_samps['a'].median().values
    incl = flat_samps['incl'].median().values
    ecc = flat_samps['ecc'].median().values
    period = flat_samps['period'].median().values
    omega = flat_samps['omega'].median().values

    print(f"Report on {TIC_TARGET}.")
    print(f"M1 has a mass: {m1:.3f} Msol, radius: {r1:.3f} Rsol, logG: {logg1:3f} and stellar type {stype1}")
    print(f"M2 has a mass: {m2:.3f} Msol, radius: {r2:.3f} Rsol, logG: {logg2:3f} and stellar type {stype2}")
    print(f"The binary system has inclination: {incl:.3f}, semi-major axis: {a:.3f} AU, and ecc: {ecc:3f}.")
    print(f"This binary system has a period of {period:.3f} days.")

    if return_dict:
        return {'m1':m1,'r1':r1,'logg1':logg1,'stype1':stype1,
                'm2':m2,'r2':r2,'logg2':logg2,'stype2':stype2,
                'a':a,'incl':incl,'ecc':ecc,'period':period, 'omega':omega}


def get_nearest_eep_from_logg(TIC_ID):


    eep_dict = {1: 'PMS',
         202: 'ZAMS',
         353: 'IAMS',
         454: 'TAMS',
         605: 'RGBTip',
         631: 'ZAHB',
         707: 'TAHB',
         808: 'TPAGB',
         1409: 'post-AGB',
         1710: 'WDCS'}
    primary_eeps = np.array([1, 202, 353, 454, 605, 631, 707, 808, 1409, 1710])

    ticparams = get_system_data_for_pymc3_model(TIC_ID.replace(' ','_').replace('-','_'))

    FEH = ticparams[0]['joker_param']['FE_H']

    sysparams = write_a_story_for_system(TIC_TARGET=TIC_ID,
                                        model_type='8x',chains=6,return_dict=True)

    print(sysparams)
    M1,LOGG1 = sysparams['m1'],sysparams['logg1']
    M2,LOGG2 = sysparams['m2'],sysparams['logg2']

    valid_eep1,valid_eep2 = [],[]

    ages = np.linspace(6, 10.12, 25000)

    from isochrones.mist import MIST_EvolutionTrack
    mtrack = MIST_EvolutionTrack()

    for iage in tqdm(ages):
        # try:
        test_eep1 = mtrack.get_eep_accurate(mass=M1, age=iage, feh=FEH)
        valid_eep1.append(test_eep1)


        try:
            test_eep2 = mtrack.get_eep_accurate(M2, iage, FEH)
            valid_eep2.append(test_eep2)
        except:
            continue 
    print(len(valid_eep1), len(valid_eep2))
    valid_logg1 = [mtrack.interp_value([M1, ee, FEH], ['logg'])[0] for ee in valid_eep1]
    valid_logg2 = [mtrack.interp_value([M2, ee, FEH], ['logg'])[0] for ee in valid_eep2]

    closest_eep1 = valid_eep1[np.argmin(abs(np.array(valid_logg1) - LOGG1))]
    closest_eep2 = valid_eep2[np.argmin(abs(np.array(valid_logg2) - LOGG2))]

    closest_state1 = np.argmin(abs(primary_eeps - closest_eep1))
    closest_state2 = np.argmin(abs(primary_eeps - closest_eep2))

    print(f'M1 appears to be in the {eep_dict[closest_state1]} state')
    print(f'M2 appears to be in the {eep_dict[closest_state2]} state')


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
    
    tess_pwd = DD+"apotess_systems/"
    current_tess_files = os.listdir(tess_pwd)
    
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



def run_with_sparse_data(x,y,yerr, sparse_factor=5, return_mask=False):
    """ quick function to take arrays and randomly downsample to reduce array
    size for computational gains. 
    """
    np.random.seed(68594)
    m = np.random.rand(len(x)) < 1.0 / sparse_factor
    x = x[m]
    y = y[m]
    yerr = yerr[m]
    if return_mask:
        return (x,y,yerr, m)
    else:
        return (x,y,yerr)


def sparse_out_eclipse_phase_curve(pymc3_model_dict,sf):

    x,y,yerr = pymc3_model_dict['x'], pymc3_model_dict['y'], pymc3_model_dict['yerr']
    lit_period, lit_t0, lit_tn = pymc3_model_dict['lit_period'], pymc3_model_dict['lit_t0'], pymc3_model_dict['lit_tn']

    dur1 = pymc3_model_dict['blsres']['duration_at_max_power'].value
    foldedx = fold(x, lit_period, lit_t0)
    inds = np.argsort(foldedx)
    # print(lk_sigma)

    durF = 3.5

    mask1 = (foldedx[inds] > 0.5*lit_period - durF*dur1) & (foldedx[inds] < 0.5*lit_period + durF*dur1)
    mask2 = (foldedx[inds] > -0.5*lit_period - durF*dur1) & (foldedx[inds] < -0.5*lit_period + durF*dur1)
    mask3 = (foldedx[inds] > -durF*dur1) & (foldedx[inds] < durF*dur1)

    cmplt_mask = ~mask1&~mask2&~mask3
    lk_sigma = np.std(y[inds][cmplt_mask])

    x1,y2,yerr2,m = run_with_sparse_data(foldedx[inds][cmplt_mask],
                                            y[inds][cmplt_mask],
                                            yerr[inds][cmplt_mask],
                                            sparse_factor=sf,return_mask=True)

    x_ = np.append(x[inds][~cmplt_mask], x[inds][cmplt_mask][m])
    y_ = np.append(y[inds][~cmplt_mask], y[inds][cmplt_mask][m])
    yerr_ = np.append(yerr[inds][~cmplt_mask], yerr[inds][cmplt_mask][m])

    x_ = np.ascontiguousarray(x_, dtype=np.float64)
    y_ = np.ascontiguousarray(y_, dtype=np.float64)
    yerr_ = np.ascontiguousarray(yerr_, dtype=np.float64)

    x_inds = np.argsort(x_)
    y_ = y_[x_inds]
    yerr_ = yerr_[x_inds]
    x_ = x_[x_inds]

    pymc3_model_dict['lk_sigma'] = lk_sigma

    if len(x) > 1.5e4:
        print("sparsifying inter-transit phase curve")
        x,y,yerr = x_, y_, yerr_
        pymc3_model_dict['x'], pymc3_model_dict['y'], pymc3_model_dict['yerr'] = x,y,yerr
    else:
        print("no need to sparsify, skipping.")

    return pymc3_model_dict


def load_all_data_for_pymc3_model(TIC_TARGET, sparse_factor=1, nsig=3, 
                                save_data_to_dict=False,
                                sparsify_phase_curve=False):
    # TIC_TARGET = 'TIC 20215452'

    res, blsres, sysapodat = get_system_data_for_pymc3_model(TIC_TARGET)

    sys_dest, fig_dest = check_for_system_directory(TIC_TARGET, return_directories=True)


        # {'mvPrior_mu':mvPrior_mu, 
        # 'mvPrior_cov':mvPrior_cov,
        # 'logm1':[np.mean(np.log(m1)), np.std(np.log(m1))],
        # 'logr1':[np.mean(np.log(r1)), np.std(np.log(r1))],
        # 'logq':[np.mean(log_q), np.std(log_q)],
        # 'logs':[np.mean(log_s), np.std(log_s)], 
        # 'logk':[np.mean(log_k), np.std(log_k)], 
        # 'mvPrior':mvPrior
        # }

    isochrones_res_dict = get_isochrones_binmod_res(TIC_TARGET, nsig=nsig, fig_dest=fig_dest)
    
    rv_time = astropy.time.Time(sysapodat['JD'], format='jd', scale='tcb')
    # print(sysapodat['MJD'])
    # texp = get_texp_from_lightcurve(res)

    texp = 0.001388888888888889  ### 2min cadence from SPOC

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

    if sparse_factor > 1 and sparsify_phase_curve is not True:
        print("sparsifying the entire lightcurve")
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
    # no_transit_lks = model_lk_data.remove_outliers(sigma=1)
    y_masked = 1000 * (no_transit_lks.flux.value / np.median(no_transit_lks.flux.value) - 1)
    lk_sigma = np.std(y_masked)
    print(lk_sigma)

    Ntrans = np.floor((x.max() - lit_t0) / lit_period)
    print(Ntrans)
    lit_tn = lit_t0  + Ntrans * lit_period

    bls2res = highres_secondary_transit_bls(res,blsres)
    ecosw_tv = estimate_ecosw(bls2res, blsres)

    compiled_dict =  {
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
        'isores' : isochrones_res_dict,
        'sys_params' : res['joker_param'],
        'blsres':blsres,
        'model_lk_data' :model_lk_data
    }

    if sparsify_phase_curve:
        print("running inter-transit phase curve sparsification")
        compiled_dict = sparse_out_eclipse_phase_curve(compiled_dict, sf=sparse_factor)

    if save_data_to_dict:
        print("saving compiled pymc3 dict to memory")
        file = open(DD + 
            f"pymc3_data_dicts/{TIC_TARGET.replace('-','_').replace(' ','_')}_sf{int(sparse_factor)}_pymc3_data_dict",'wb')
        pk.dump(compiled_dict, file)
        file.close()

        
    return compiled_dict


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
    fig = plt.figure(figsize=(12,18), constrained_layout=False)

    # fig, ax4 = plt.subplots(figsize=(12,18))

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
    if 'log_sigma_rv' not in list(soln.keys()):
        lsig_rv = soln['log_sigma_rv_upperbound__']
    else:
        lsig_rv = soln['log_sigma_rv']
    ax4.errorbar(x_rv_fold, y_rv-mean, yerr=np.sqrt(np.exp(2.*lsig_rv) + yerr_rv**2.),
                 fmt='.',c='black',ecolor='red', label='RV obs')
    ax4.set_title(title)
    
    y1,y2 = ax4.set_ylim()
    x1,x2 = ax4.set_xlim()
    ax4.vlines(0.25*(x2-x1) + x1,
               0.25*(y2-y1) + y1,
               0.25*(y2-y1) + y1 + np.exp(lsig_rv)
              )
    filename = filename + '.png'
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    if RETURN_FILENAME:
        return filename


def make_folded_lightcurve_from_blsres(TICID):
    
    file = open(f"/Users/karljaehnig/CCA_work/GAT/joker_TESS_lightcurve_files/{TICID.replace(' ','_').replace('-','_')}_highres_bls_params.pickle",'rb')
    blsres = pk.load(file)
    file.close()

    file = open(f"/Users/karljaehnig/CCA_work/GAT/joker_TESS_lightcurve_files/{TICID.replace(' ','_').replace('-','_')}_lightcurve_data.pickle","rb")
    res = pk.load(file)
    file.close()
    
        #     print(calibverr.info)
    # Grab cross-match IDs
    sysapodat = allvis17[allvis17['APOGEE_ID'] == res['joker_param']['APOGEE_ID']]

    ## joining calib RV_ERRs with the RVs
    sysapodat = astab.join(sysapodat, calibverr['VISIT_ID','CALIB_VERR'], keys=('VISIT_ID','VISIT_ID'))

    
    
    lks = res['lk_coll'].stitch(corrector_func=lambda x: x.remove_nans().normalize())
#     print([len(ii) for ii in blsres['period']])
#     print('sectors: ',res['lk_coll'].sector,'\n'\
#           'bls_per(s): ',blsres['period'],'\n'\
#           'MAP_P: ',res['joker_param']['MAP_P'])

    map_period = res['joker_param']['MAP_P']
    map_t0_bmjd = res['joker_param']['MAP_t0_bmjd']
    map_t0 = astropy.time.Time(map_t0_bmjd, format='mjd', scale='tcb')

    bls_period = blsres['period_at_max_power'].value
    bls_t0 = blsres['t0_at_max_power'].value
        
    rv_time = astropy.time.Time(sysapodat['MJD'], format='mjd', scale='tcb')

    abs_time_vmin = 0.0
    abs_time_vmax = max(lks.time.btjd.max()-lks.time.btjd.min(), rv_time.btjd.max()-rv_time.btjd.min())
#     print(abs_time_vmin, abs_time_vmax)
    
    fig,ax = plt.subplots(figsize=(16,20), nrows=4, ncols=2)
#     ax[-1,-1].remove()
    
    fig.text(0.5,0.885,f'un-folded observations (TESS --- APOGEE) [{TICID}]',
             fontdict={'horizontalalignment':'center','fontweight':'bold','fontsize':14})
    lc_unfolded = ax[0,0].scatter(lks.time.value, lks.flux.value,marker='o', s=0.5,
                              c=lks.time.value - lks.time.min().value, 
                                  vmin=abs_time_vmin, vmax=abs_time_vmax, cmap=cm.inferno)

    rv_unfolded = ax[0,1].plot(rv_time.btjd, 
                               sysapodat['VHELIO'] - res['joker_param']['MAP_v0'],
                               marker='o',ls='None',mec='black')
    
    fig.text(0.5,0.69,'bls folded observations (TESS --- APOGEE)',
             fontdict={'horizontalalignment':'center','fontweight':'bold','fontsize':14})
    
    Nbins = 100  # int(len(lks.time.value) / 1000.)
    bins = np.linspace(-0.5*bls_period, 0.5*bls_period, Nbins)
    x_ = fold(lks.time.value, bls_period, bls_t0)
    y_ = lks.flux.value
    num, _ = np.histogram(x_, bins, weights=y_)
    denom, _ = np.histogram(x_, bins)
    num[denom > 0] /= denom[denom > 0]
    num[denom == 0] = np.nan
#     def running_mean(x, N):
#         cumsum = np.cumsum(np.insert(x, 0, 0)) 
#         return (cumsum[N:] - cumsum[:-N]) / float(N)
    
#     folded_lcx = fold(lks.time.value, bls_period, bls_t0)
#     inds = np.argsort(folded_lcx)
    
#     folded_rmy = running_mean(lks.flux.value[inds],Nbins)
    
#     rmx = np.linspace(folded_lcx.min(), folded_lcx.max(), len(folded_rmy))
#     folded_rmx = fold(rmx, bls_period, bls_t0)
    
    lc_folded_bls = ax[1,0].scatter(fold(lks.time.value, bls_period, bls_t0), 
               lks.flux.value, marker='o',s=0.5,
               c=lks.time.value - lks.time.min().value,
               cmap=cm.inferno, vmin=abs_time_vmin, vmax=abs_time_vmax#fold(lks.time.value, bls_period, bls_t0)
                )
    ax[1,0].plot(0.5 * (bins[1:] + bins[:-1]), num, color='white', lw=6,zorder=10)
    ax[1,0].plot(0.5 * (bins[1:] + bins[:-1]), num, color='cyan', lw=3,zorder=11)
#     ax[1,0].set_ylim(lks.flux.min(), lks.flux.max())
    
    rv_folded_bls = ax[1,1].scatter(fold(sysapodat['MJD'].value, bls_period, blsres['t0_at_max_power'].mjd),
                                    sysapodat['VHELIO']-res['joker_param']['MAP_v0'],
                                    marker='o',c=sysapodat['MJD'].value-min(sysapodat['MJD']),
                                    cmap=cm.inferno, vmin=abs_time_vmin, vmax=abs_time_vmax,
                                    ec='black')
    ax[1,1].axvline(0.0,ls='--')
    ax[1,1].axhline(0.0,ls='--')

    fig.text(0.5,0.49,'MAP folded observations (TESS --- APOGEE)',
             fontdict={'horizontalalignment':'center','fontweight':'bold','fontsize':14})
    lc_folded_map = ax[2,0].scatter(fold(lks.time.value, map_period, map_t0.btjd), 
               lks.flux.value, marker='o',s=0.5,
               c=lks.time.value - lks.time.min().value,
               cmap=cm.inferno, vmin=abs_time_vmin, vmax=abs_time_vmax#fold(lks.time.value, bls_period, bls_t0)
                )
    
    rv_folded_map = ax[2,1].scatter(fold(sysapodat['MJD'].value, map_period, map_t0.mjd),
                                    sysapodat['VHELIO']-res['joker_param']['MAP_v0'],
                                    marker='o',c=sysapodat['MJD'].value-min(sysapodat['MJD']),
                                    cmap=cm.inferno, vmin=abs_time_vmin, vmax=abs_time_vmax,
                                    ec='black')
    fig.colorbar(rv_folded_map, ax=ax[1:3,1],shrink=1.0, pad=0.01, fraction=0.05,label='days')
    ax[2,1].axvline(0.0,ls='--')
    ax[2,1].axhline(0.0,ls='--')
    
    ax[3,0].set_title("GAIA CMD Target Location", fontsize=14, fontweight='bold')
    allstar_Gmag = allstar17['phot_g_mean_mag'] - (5.*np.log10(1000./allstar17['parallax']) - 5.)
    unimodal_Gmag = hq_jk_allstar_tess_edr3['phot_g_mean_mag'] - (5.*np.log10(1000./hq_jk_allstar_tess_edr3['parallax']) - 5)
    tess_obs_Gmag = hq_jk_allstar_tess_edr3_w_tess_obs['phot_g_mean_mag'] - (
                        5.*np.log10(1000./hq_jk_allstar_tess_edr3_w_tess_obs['parallax']) - 5)
    target_Gmag = res['joker_param']['phot_g_mean_mag'] - (5.*np.log10(1000./res['joker_param']['parallax']) - 5.)
    
    ax[3,0].plot(allstar17['bp_rp'], allstar_Gmag,
                 marker=',',color='gray', ls='None',label='allstar')
    ax[3,0].plot(hq_jk_allstar_tess_edr3['bp_rp'],unimodal_Gmag,
                 marker='.',color='tab:red',ls='None', label='unimodal')
    ax[3,0].plot(hq_jk_allstar_tess_edr3_w_tess_obs['bp_rp'], tess_obs_Gmag,
                 marker='^',ms=8,color='black',ls='None',mfc='None',label='unimodal w TESS')
    ax[3,0].plot(res['joker_param']['bp_rp'],target_Gmag,
                 marker='^',ms=8,color='tab:blue',lw=2,ls='None',label=TICID)
    ax[3,0].set_ylim(15,-15)
    ax[3,0].set_xlim(-1,7)
    ax[3,0].legend(fontsize=10)
    
    joker_param = hq_jk_allstar_tess_edr3.to_pandas()[hq_jk_allstar_tess_edr3['APOGEE_ID'] == res['joker_param']['APOGEE_ID']]
#     print(joker_param['phot_g_mean_mag'])
    param_str = (f"Gaia G: {joker_param['phot_g_mean_mag'].squeeze()}\nTeff: {int(joker_param['TEFF'].squeeze())}\nLogG: {joker_param['LOGG'].squeeze()}\nM_H: {joker_param['M_H'].squeeze()}\necc: {joker_param['MAP_e'].squeeze()}\nMAP_P: {joker_param['MAP_P'].squeeze()}\nBLS_P: {bls_period}")
    ax[3,1].scatter(0,0,ec='None',fc='None',label=param_str)
    ax[3,1].legend(loc='center',scatterpoints=0, fontsize=18, frameon=False)
    
#     fig.colorbar(lc_folded_)
#     ax.set_xlim(-1,1)


def get_all_transit_params(TIC_ID, jk_row):
    
    res = {'periods':[],
          'durations':[],
          't0s':[],
          'depths':[]}
    
    period_grid = np.exp(np.linspace(np.log(0.5*jk_row['MAP_P']),
                                     np.log(2.*jk_row['MAP_P']),
                                     10000)).squeeze()
    
#     period_grid = np.exp(np.linspace(np.log(0.1), np.log(100),1000))
    dur_grid = np.exp(np.linspace(np.log(0.001),np.log(0.09),50))

    npts = 5000
    pmin = period_grid.min()
    pmax = period_grid.max()
    mindur = dur_grid.min()

    print("Downloading all available TESS data.")
    lk_search = lk.search_lightcurve(TIC_ID,
             mission='TESS',
            cadence='short',
            author='SPOC'
             )
    
    unprocessed_lkcoll = lk_search.download_all(quality_bitmask='hardest')
    all_lks = unprocessed_lkcoll.stitch(corrector_func=lambda x: x.remove_nans().normalize().flatten())
    
    print("Separating TESS sector data into groups.")
    _, __, grp_ind = get_multiple_ranges(unprocessed_lkcoll)
    
    for ii,ind in enumerate(grp_ind):
        
        print(f"Running BLS on group {ii}, sectors: {unprocessed_lkcoll[ind].sector}")
        lkgrp = unprocessed_lkcoll[ind].stitch(corrector_func=lambda x: x.remove_nans().normalize().flatten())
        
        maxtime = lkgrp.time.max().value
        mintime = lkgrp.time.min().value

        freq_f = int( ((pmin**-1 - pmax**-1) * (maxtime - mintime)**2) / (npts * mindur) ) 
        
        lkgrpBLS = lkgrp.to_periodogram('bls',
                                period=period_grid,
                                frequency_factor = freq_f, duration=dur_grid)
        
        res['periods'].append(lkgrpBLS.period_at_max_power.value)
        res['t0s'].append(lkgrpBLS.transit_time_at_max_power)
        res['durations'].append(lkgrpBLS.duration_at_max_power.value)
        res['depths'].append(lkgrpBLS.depth_at_max_power)
        
    print("Finished.")
    res['unprocessed_lk_coll'] = unprocessed_lkcoll
    res['all_lks'] = all_lks
    res['period_linspace'] = [0.5*jk_row['MAP_P'], 2.*jk_row['MAP_P'], len(period_grid)]
    res['dur_linspace'] = [0.001,0.09, len(dur_grid)]
    res['freq_factor'] = freq_f
    
    return res


def make_folded_lightcurve_from_blsres(TICID):
    
    file = open(f"/Users/karljaehnig/CCA_work/GAT/joker_TESS_lightcurve_files/{TICID.replace(' ','_').replace('-','_')}_highres_bls_params.pickle",'rb')
    blsres = pk.load(file)
    file.close()

    file = open(f"/Users/karljaehnig/CCA_work/GAT/joker_TESS_lightcurve_files/{TICID.replace(' ','_').replace('-','_')}_lightcurve_data.pickle","rb")
    res = pk.load(file)
    file.close()
    
        #     print(calibverr.info)
    # Grab cross-match IDs
    sysapodat = allvis17[allvis17['APOGEE_ID'] == res['joker_param']['APOGEE_ID']]

    ## joining calib RV_ERRs with the RVs
    sysapodat = astab.join(sysapodat, calibverr['VISIT_ID','CALIB_VERR'], keys=('VISIT_ID','VISIT_ID'))

    
    
    lks = res['lk_coll'].stitch(corrector_func=lambda x: x.remove_nans().normalize())
#     print([len(ii) for ii in blsres['period']])
#     print('sectors: ',res['lk_coll'].sector,'\n'\
#           'bls_per(s): ',blsres['period'],'\n'\
#           'MAP_P: ',res['joker_param']['MAP_P'])

    map_period = res['joker_param']['MAP_P']
    map_t0_bmjd = res['joker_param']['MAP_t0_bmjd']
    map_t0 = astropy.time.Time(map_t0_bmjd, format='mjd', scale='tcb')

    bls_period = blsres['period_at_max_power'].value
    bls_t0 = blsres['t0_at_max_power'].value
        
    rv_time = astropy.time.Time(sysapodat['MJD'], format='mjd', scale='tcb')

    abs_time_vmin = 0.0
    abs_time_vmax = max(lks.time.btjd.max()-lks.time.btjd.min(), rv_time.btjd.max()-rv_time.btjd.min())
#     print(abs_time_vmin, abs_time_vmax)
    
    fig,ax = plt.subplots(figsize=(16,20), nrows=4, ncols=2)
#     ax[-1,-1].remove()
    
    fig.text(0.5,0.885,f'un-folded observations (TESS --- APOGEE) [{TICID}]',
             fontdict={'horizontalalignment':'center','fontweight':'bold','fontsize':14})
    lc_unfolded = ax[0,0].scatter(lks.time.value, lks.flux.value,marker='o', s=0.5,
                              c=lks.time.value - lks.time.min().value, 
                                  vmin=abs_time_vmin, vmax=abs_time_vmax, cmap=cm.inferno)

    rv_unfolded = ax[0,1].plot(rv_time.btjd, 
                               sysapodat['VHELIO'] - res['joker_param']['MAP_v0'],
                               marker='o',ls='None',mec='black')
    
    fig.text(0.5,0.69,'bls folded observations (TESS --- APOGEE)',
             fontdict={'horizontalalignment':'center','fontweight':'bold','fontsize':14})
    
    Nbins = 100  # int(len(lks.time.value) / 1000.)
    bins = np.linspace(-0.5*bls_period, 0.5*bls_period, Nbins)
    x_ = fold(lks.time.value, bls_period, bls_t0)
    y_ = lks.flux.value
    num, _ = np.histogram(x_, bins, weights=y_)
    denom, _ = np.histogram(x_, bins)
    num[denom > 0] /= denom[denom > 0]
    num[denom == 0] = np.nan
#     def running_mean(x, N):
#         cumsum = np.cumsum(np.insert(x, 0, 0)) 
#         return (cumsum[N:] - cumsum[:-N]) / float(N)
    
#     folded_lcx = fold(lks.time.value, bls_period, bls_t0)
#     inds = np.argsort(folded_lcx)
    
#     folded_rmy = running_mean(lks.flux.value[inds],Nbins)
    
#     rmx = np.linspace(folded_lcx.min(), folded_lcx.max(), len(folded_rmy))
#     folded_rmx = fold(rmx, bls_period, bls_t0)
    
    lc_folded_bls = ax[1,0].scatter(fold(lks.time.value, bls_period, bls_t0), 
               lks.flux.value, marker='o',s=0.5,
               c=lks.time.value - lks.time.min().value,
               cmap=cm.inferno, vmin=abs_time_vmin, vmax=abs_time_vmax#fold(lks.time.value, bls_period, bls_t0)
                )
    ax[1,0].plot(0.5 * (bins[1:] + bins[:-1]), num, color='white', lw=6,zorder=10)
    ax[1,0].plot(0.5 * (bins[1:] + bins[:-1]), num, color='cyan', lw=3,zorder=11)
#     ax[1,0].set_ylim(lks.flux.min(), lks.flux.max())
    
    rv_folded_bls = ax[1,1].scatter(fold(sysapodat['MJD'].value, bls_period, blsres['t0_at_max_power'].mjd),
                                    sysapodat['VHELIO']-res['joker_param']['MAP_v0'],
                                    marker='o',c=sysapodat['MJD'].value-min(sysapodat['MJD']),
                                    cmap=cm.inferno, vmin=abs_time_vmin, vmax=abs_time_vmax,
                                    ec='black')
    ax[1,1].axvline(0.0,ls='--')
    ax[1,1].axhline(0.0,ls='--')

    fig.text(0.5,0.49,'MAP folded observations (TESS --- APOGEE)',
             fontdict={'horizontalalignment':'center','fontweight':'bold','fontsize':14})
    lc_folded_map = ax[2,0].scatter(fold(lks.time.value, map_period, map_t0.btjd), 
               lks.flux.value, marker='o',s=0.5,
               c=lks.time.value - lks.time.min().value,
               cmap=cm.inferno, vmin=abs_time_vmin, vmax=abs_time_vmax#fold(lks.time.value, bls_period, bls_t0)
                )
    
    rv_folded_map = ax[2,1].scatter(fold(sysapodat['MJD'].value, map_period, map_t0.mjd),
                                    sysapodat['VHELIO']-res['joker_param']['MAP_v0'],
                                    marker='o',c=sysapodat['MJD'].value-min(sysapodat['MJD']),
                                    cmap=cm.inferno, vmin=abs_time_vmin, vmax=abs_time_vmax,
                                    ec='black')
    fig.colorbar(rv_folded_map, ax=ax[1:3,1],shrink=1.0, pad=0.01, fraction=0.05,label='days')
    ax[2,1].axvline(0.0,ls='--')
    ax[2,1].axhline(0.0,ls='--')
    
    ax[3,0].set_title("GAIA CMD Target Location", fontsize=14, fontweight='bold')
    allstar_Gmag = allstar17['phot_g_mean_mag'] - (5.*np.log10(1000./allstar17['parallax']) - 5.)
    unimodal_Gmag = hq_jk_allstar_tess_edr3['phot_g_mean_mag'] - (5.*np.log10(1000./hq_jk_allstar_tess_edr3['parallax']) - 5)
    tess_obs_Gmag = hq_jk_allstar_tess_edr3_w_tess_obs['phot_g_mean_mag'] - (
                        5.*np.log10(1000./hq_jk_allstar_tess_edr3_w_tess_obs['parallax']) - 5)
    target_Gmag = res['joker_param']['phot_g_mean_mag'] - (5.*np.log10(1000./res['joker_param']['parallax']) - 5.)
    
    ax[3,0].plot(allstar17['bp_rp'], allstar_Gmag,
                 marker=',',color='gray', ls='None',label='allstar')
    ax[3,0].plot(hq_jk_allstar_tess_edr3['bp_rp'],unimodal_Gmag,
                 marker='.',color='tab:red',ls='None', label='unimodal')
    ax[3,0].plot(hq_jk_allstar_tess_edr3_w_tess_obs['bp_rp'], tess_obs_Gmag,
                 marker='^',ms=8,color='black',ls='None',mfc='None',label='unimodal w TESS')
    ax[3,0].plot(res['joker_param']['bp_rp'],target_Gmag,
                 marker='^',ms=8,color='tab:blue',lw=2,ls='None',label=TICID)
    ax[3,0].set_ylim(15,-15)
    ax[3,0].set_xlim(-1,7)
    ax[3,0].legend(fontsize=10)
    
    joker_param = hq_jk_allstar_tess_edr3.to_pandas()[hq_jk_allstar_tess_edr3['APOGEE_ID'] == res['joker_param']['APOGEE_ID']]
#     print(joker_param['phot_g_mean_mag'])
    param_str = (f"Gaia G: {joker_param['phot_g_mean_mag'].squeeze()}\nTeff: {int(joker_param['TEFF'].squeeze())}\nLogG: {joker_param['LOGG'].squeeze()}\nM_H: {joker_param['M_H'].squeeze()}\necc: {joker_param['MAP_e'].squeeze()}\nMAP_P: {joker_param['MAP_P'].squeeze()}\nBLS_P: {bls_period}")
    ax[3,1].scatter(0,0,ec='None',fc='None',label=param_str)
    ax[3,1].legend(loc='center',scatterpoints=0, fontsize=18, frameon=False)
    
#     fig.colorbar(lc_folded_)
#     ax.set_xlim(-1,1)


def folded_bin_histogram(Nbins=100, lks=None, bls_period=None, bls_t0=None):
    Nbins = Nbins  # int(len(lks.time.value) / 1000.)
    bins = np.linspace(-0.5*bls_period, 0.5*bls_period, Nbins)
    x_ = fold(lks.time.value, bls_period, bls_t0)
    y_ = lks.flux.value
    num, _ = np.histogram(x_, bins, weights=y_)
    denom, _ = np.histogram(x_, bins)
    num[denom > 0] /= denom[denom > 0]
    num[denom == 0] = np.nan
#     def running_mean(x, N):
#         cumsum = np.cumsum(np.insert(x, 0, 0)) 
#         return (cumsum[N:] - cumsum[:-N]) / float(N)
    
#     folded_lcx = fold(lks.time.value, bls_period, bls_t0)
#     inds = np.argsort(folded_lcx)
    
#     folded_rmy = running_mean(lks.flux.value[inds],Nbins)
    
#     rmx = np.linspace(folded_lcx.min(), folded_lcx.max(), len(folded_rmy))
#     folded_rmx = fold(rmx, bls_period, bls_t0)
    

    return (0.5 * (bins[1:] + bins[:-1]),
            num)


