import lightkurve as lk
import astropy.table as astab
import pandas as pd
import numpy as np
import astropy
import pickle as pk
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, TextBox

from tqdm import tqdm
# %pylab inline
# pylab.rcParams['figure.figsize'] = (16, 8)
import warnings
import astropy.table as astab
from astropy.io import fits

warnings.filterwarnings('ignore',
    message="WARNING (theano.tensor.opt): Cannot construct a scalar test value from a test value with no size:"
)
print(astropy.__version__)

import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
import exoplanet as xo

import arviz as az
from corner import corner

from scipy.signal import savgol_filter

# %matplotlib widget
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


allvis17 = astab.Table.read("/Users/kjaehnig/CCA_work/GAT/dr17_joker/allVisit-dr17-synspec.fits",hdu=1, format='fits')
allstar17 = astab.Table.read("/Users/kjaehnig/CCA_work/GAT/dr17_joker/allStar-dr17-synspec-gaiaedr3-xm.fits")
allstar17 = allstar17[(allstar17['bp_rp'] < 10) & (allstar17['phot_g_mean_mag'] < 25)]
calibverr = astab.Table.read(dd+'dr17_joker/allVisit-dr17-synspec-min3-calibverr.fits', format='fits', hdu=1)




def fold(x, period, t0):
    hp = 0.5 * period
    return (x - t0 + hp) % period - hp

def get_texp_from_lightcurve(res):
    with fits.open(res['all_lks'].filename) as hdu:
        hdr = hdu[1].header

    texp = hdr["FRAMETIM"] * hdr["NUM_FRM"]
    texp /= 60.0 * 60.0 * 24.0
    print(texp, texp*60*60*24)

    return texp


def get_system_data_for_pymc3_model(TICID):
    
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


TIC_TARGET = 'TIC 272074664'

res, blsres, sysapodat = get_system_data_for_pymc3_model(TIC_TARGET)

rv_time = astropy.time.Time(sysapodat['MJD'], format='mjd', scale='tcb')

texp = get_texp_from_lightcurve(res)

x_rv = rv_time.btjd
y_rv = sysapodat['VHELIO'] - res['joker_param']['MAP_v0']
yerr_rv = sysapodat['CALIB_VERR']


x =    res['all_lks'].remove_nans().time.btjd
y =    res['all_lks'].remove_nans().flux.value
yerr = res['all_lks'].remove_nans().flux_err.value

x_lk_ref = min(x)

x_rv = x_rv - x_lk_ref

x = x - x_lk_ref 
y = (y / np.median(y) - 1)

y *= 1e3


def run_with_sparse_data(x,y,yerr, use_sparse_data=False):
    if use_sparse_data:
        np.random.seed(68594)
        m = np.random.rand(len(x)) < 1.0 / 15
        x = x[m]
        y = y[m]
        yerr = yerr[m]
    return x,y,yerr

x,y,yerr = run_with_sparse_data(x,y,yerr,False)


x = np.ascontiguousarray(x, dtype=np.float64)
y = np.ascontiguousarray(y, dtype=np.float64)
yerr = np.ascontiguousarray(yerr, dtype=np.float64)


x_rv = np.ascontiguousarray(x_rv, dtype=np.float64)
y_rv = np.ascontiguousarray(y_rv, dtype=np.float64)
yerr_rv = np.ascontiguousarray(yerr_rv, dtype=np.float64)

bls_period = blsres['period_at_max_power'].value
bls_t0 = blsres['t0_at_max_power'].btjd - x_lk_ref
print('lightcurve N datapoints: ',len(x),len(y),len(yerr), 'transit_epoch: ',bls_t0)


lit_period = bls_period  #bls_period      ### THESE ARE THE TWO VARIABLES USED
lit_t0 = bls_t0   #bls_t0             ### IN THE PYMC3 MODEL BELOW


transit_mask = res['all_lks'].create_transit_mask(
    period=blsres['period_at_max_power'].value,
    duration=5.*blsres['duration_at_max_power'].value,
    transit_time=blsres['t0_at_max_power']
)

no_transit_lks = res['all_lks'][~transit_mask]
y_masked = 1000 * (no_transit_lks.flux.value / np.median(no_transit_lks.flux.value) - 1)
lk_sigma = np.mean(y_masked)



t = np.linspace(x_rv.min(), x_rv.max(), 5000)
tlc = np.linspace(x.min(), x.max(), 5000)

rvK = xo.estimate_semi_amplitude(bls_period, x_rv, y_rv, yerr_rv, t0s=bls_t0)[0]
print(rvK)



def build_model(mask=None, start=None,
    LC_GP_PARAMS = [[0.1,0.5],
                    [1.0,10.0],
                    [1.0,10.0]]):
    if mask is None:
        mask = np.ones(len(x), dtype='bool')
    with pm.Model() as model:

        # Systemic parameters
        mean_lc = pm.Normal("mean_lc", mu=0.0, sd=10.0)
        mean_rv = pm.Normal("mean_rv", mu=0.0, sd=50.0)

        u1 = xo.QuadLimbDark("u1")
        u2 = xo.QuadLimbDark("u2")

        # Parameters describing the primary
        log_M1 = pm.Normal("log_M1", mu=0.0, sigma=10.0)
#         log_R1 = pm.Uniform('log_R1', lower=np.log(1e-5), upper=np.log(1000))
        log_R1 = pm.Normal("log_R1", mu=0.0, sigma=10.0)
        M1 = pm.Deterministic("M1", tt.exp(log_M1))
        R1 = pm.Deterministic("R1", tt.exp(log_R1))

        # Secondary ratios
        log_k = pm.Normal("log_k", mu=0.0, sigma=5.0)  # radius ratio
        
        logK = pm.Normal("logK", mu=np.log(rvK), sigma=5.0, testval=np.log(rvK)) 
        
        log_q = pm.Normal("log_q", mu=0.0, sigma=5.0)  # mass ratio
        log_s = pm.Normal("log_s", mu=0.0, sigma=10.0)  # surface brightness ratio
        pm.Deterministic("k", tt.exp(log_k))
        pm.Deterministic("q", tt.exp(log_q))
        pm.Deterministic("s", tt.exp(log_s))

        # Prior on flux ratio
        pm.Normal(
            "flux_prior",
            mu=0.5,
            sigma=0.25,
            observed=tt.exp(2 * log_k + log_s),
        )

        # Parameters describing the orbit
        b = xo.ImpactParameter("b", ror=tt.exp(log_k), testval=0.9)

            
        log_period = pm.Normal("log_period", mu=np.log(lit_period), sigma=5.0)
        period = pm.Deterministic("period", tt.exp(log_period))
        t0 = pm.Normal("t0", mu=lit_t0, sigma=1.0)

        # Parameters describing the eccentricity: ecs = [e * cos(w), e * sin(w)]
        ecs = pmx.UnitDisk("ecs", testval=np.array([0.0, 1e-5]))
        ecc = pm.Deterministic("ecc", tt.sqrt(tt.sum(ecs ** 2)))
        omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))

        # Build the orbit
        R2 = pm.Deterministic("R2", tt.exp(log_k + log_R1))
        M2 = pm.Deterministic("M2", tt.exp(log_q + log_M1))
        orbit = xo.orbits.KeplerianOrbit(
            period=period,
            t0=t0,
            ecc=ecc,
            omega=omega,
            b=b,
            r_star=R1,
            m_star=M1,
            m_planet=M2,
        )

        # Track some other orbital elements
        pm.Deterministic("incl", orbit.incl)
        pm.Deterministic("a", orbit.a)
        
        
        
        
        # Noise model for the light curve
        
        
        slc_a, slc_b = LC_GP_PARAMS[0]
        sigma_lc = pm.InverseGamma(
            "sigma_lc",
            testval= np.mean(yerr),
            **pmx.estimate_inverse_gamma_parameters(slc_a, slc_b)
        )
        
        sgp_a, sgp_b = LC_GP_PARAMS[1]
        sigma_gp = pm.InverseGamma(
            "sigma_gp",
            testval= lk_sigma,
            **pmx.estimate_inverse_gamma_parameters(sgp_a, sgp_b),
        )

        rgp_a, rgp_b = LC_GP_PARAMS[2]
        rho_gp = pm.InverseGamma(
            "rho_gp",
            testval= 0.10 * bls_period,
            **pmx.estimate_inverse_gamma_parameters(rgp_a, rgp_b)
        )
        kernel_lc = terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, Q=1.0 / 3.)


        # Set up the light curve model
        lc = xo.SecondaryEclipseLightCurve(u1, u2, tt.exp(log_s))

        def model_lc(t):
            return (
                mean_lc
                + 1e3
                * lc.get_light_curve(orbit=orbit, r=R2, t=t, texp=texp)[:,0]
            )

        pm.Deterministic(
            "lc_pred",
            model_lc(x)
        )
        
        # Condition the light curve model on the data
        gp_lc = GaussianProcess(kernel_lc, t=x[mask], yerr=sigma_lc)
        gp_lc.marginal("obs_lc", observed=y[mask] - model_lc(x[mask]))


#         # Set up the radial velocity model


        log_sigma_rv = pm.Normal(
            "log_sigma_rv", mu=np.log(np.median(yerr_rv)), sd=10.0
        )
            
        def model_rv(t):
            return orbit.get_radial_velocity(t, K=tt.exp(logK)) + mean_rv
            
        rv_model = model_rv(x_rv)
        
        err = tt.sqrt(yerr_rv**2. + tt.exp(2*log_sigma_rv))
        
        pm.Normal("obs",mu=rv_model, sd=err, observed=y_rv)

        # Optimize the logp
        if start is None:
            start = model.test_point

            
        extras = dict(
            x=x[mask],
            y=y[mask],
            x_rv = x_rv,
            y_rv = y_rv,
            model_lc=model_lc,
            model_rv=model_rv,
            gp_lc_pred=gp_lc.predict(y[mask] - model_lc(x[mask])),
        )
        
        
        # First the RV parameters
        opti_logp = []
        # map_soln, info_ = pmx.optimize(start , return_info=True)



    return model, [], extras, start, opti_logp



def interactive_pymc3_GP_plot_widget(x,y, lit_period, lit_t0):

    t_lc_pred = np.linspace(x.min(), x.max(), 1000)

    fig = plt.figure(figsize=(10,8),constrained_layout=False)
    ax_grid = GridSpec(nrows=1000, ncols=1000)



    # slcB_btn = fig.add_subplot(ax_grid[900: ,45:70])

    # sgpA_btn = fig.add_subplot(ax_grid[900: ,90:115])
    # sgpB_btn = fig.add_subplot(ax_grid[900: ,135:160])

    # rgpA_btn = fig.add_subplot(ax_grid[900: ,180:205])
    # rgpB_btn = fig.add_subplot(ax_grid[900: ,225:250])

    unfolded_plt = fig.add_subplot(ax_grid[:450, 300:])
    folded_plt = fig.add_subplot(ax_grid[550:, 300:])



    unfolded_dat, = unfolded_plt.plot(
                        x, y, marker=',', color='black',
                        zorder=0, ls='None',
                        rasterized=True)

    folded_dat, = folded_plt.plot(
                    fold(x, lit_period, lit_t0),
                    y, marker=',', color='black',
                    zorder=0, ls='None',
                    rasterized=True)


    unfolded_med, = unfolded_plt.plot(
                        no_transit_lks.time.value-x_lk_ref,
                        no_transit_lks.flux.value,
                        marker='o',ms=2,color='tab:red',
                        zorder=1, ls='None',
                        rasterized=True)

    folded_med, = folded_plt.plot(
                        fold(no_transit_lks.time.value-x_lk_ref, lit_period, lit_t0),
                        no_transit_lks.flux.value,
                        marker='o',ms=2,color='tab:red',
                        zorder=1, ls='None',
                        rasterized=True)


    exec_btn_ax = fig.add_subplot(ax_grid[900: ,0:250 ])
    exec_btn = Button(exec_btn_ax,"Execute",
                        color='springgreen',
                        hovercolor='palegreen')


    ax_slcA = fig.add_subplot(ax_grid[0:800, 0:25  ])
    sclA_sli = Slider(ax=ax_slcA,
                        label=r'$\sigma_{LC} \alpha$',
                        valmin=0.05,
                        valmax=2.0,
                        valinit=0.1,
                        valfmt='%0.2f',
                        orientation='vertical')


    ax_slcB = fig.add_subplot(ax_grid[0:800, 45:70])
    sclB_sli = Slider(ax=ax_slcB,
                        label=r"$\sigma_{LC} \beta$",
                        valmin=1.0,
                        valmax=25.0,
                        valinit=5.0,
                        valfmt="%i",
                        orientation='vertical')


    ax_sgpA = fig.add_subplot(ax_grid[0:800, 90:115 ])
    sgpA_sli = Slider(ax=ax_sgpA,
                        label=r'$\sigma_{GP} \alpha$',
                        valmin=0.05,
                        valmax=5.0,
                        valinit=1.0,
                        valfmt='%.2f',
                        orientation='vertical')


    ax_sgpB = fig.add_subplot(ax_grid[0:800, 135:160])
    sgpB_sli = Slider(ax=ax_sgpB,
                        label=r"$\sigma_{GP} \beta$",
                        valmin=1.0,
                        valmax=25.0,
                        valinit=10.0,
                        valfmt='%i',
                        orientation='vertical')


    ax_rgpA = fig.add_subplot(ax_grid[0:800, 180:205 ])
    rgpA_sli = Slider(ax=ax_rgpA,
                        label=r'$\rho_{GP} \alpha$',
                        valmin=0.05,
                        valmax=5.0,
                        valinit=1.0,
                        valfmt='%.2f',
                        orientation='vertical')


    ax_rgpB = fig.add_subplot(ax_grid[0:800, 225:250])
    rgpB_sli = Slider(ax=ax_rgpB,
                        label=r'$\rho_{GP} \beta$',
                        valmin=1.0,
                        valmax=25.0,
                        valinit=10.0,
                        valfmt='%i',
                        orientation='vertical')

    def load_lc_gps_param():

        slcA = sclA_sli.valinit
        slcB = sclB_sli.valinit
        sgpA = sgpA_sli.valinit
        sgpB = sgpB_sli.valinit
        rgpA = rgpA_sli.valinit
        rgpB = rgpB_sli.valinit 

        model, map_soln, extras, start, opti_logp =\
            build_model(LC_GP_PARAMS=[[slcA,slcB],
                                    [sgpA,sgpB],
                                    [rgpA,rgpB]])

        t0 = start['t0']
        period = np.exp(start['log_period'])
        x_phase = np.linspace(-0.5*period, 0.5*period,1000)

        with model:
            gp_pred = (
                pmx.eval_in_model(extras['gp_lc_pred'],start) + 
                start['mean_lc'])

            lc=(pmx.eval_in_model(extras['model_lc'](t_lc_pred),start) -
                start['mean_lc'])

        xfold = fold(t_lc_pred, period, t0)
        inds = np.argsort(xfold)

        y_gp_pred = gp_pred 
        ylc = lc
        xfold = xfold[inds]
        foldedyvals = extras['y'][inds] - gp_pred[inds]

        return (y_gp_pred,ylc, xfold, foldedyvals)

    mody, ylc, fmodx, fmody = load_lc_gps_param()

    gp_unfolded, = unfolded_plt.plot(x,
                                mody,
                                ls='-',c='C0',
                                zorder=2)

    gp_folded, = folded_plt.plot(fmodx, fmody,
                            ls='-', c='C0',
                            zorder=2)


    def load_new_lc_gps_param():

        slcA = sclA_sli.val
        slcB = sclB_sli.val
        sgpA = sgpA_sli.val
        sgpB = sgpB_sli.val
        rgpA = rgpA_sli.val
        rgpB = rgpB_sli.val 

        model, map_soln, extras, start, opti_logp =\
            build_model(LC_GP_PARAMS=[[slcA,slcB],
                                    [sgpA,sgpB],
                                    [rgpA,rgpB]])

        t0 = start['t0']
        period = np.exp(start['log_period'])
        x_phase = np.linspace(-0.5*period, 0.5*period,1000)

        with model:
            gp_pred = (
                pmx.eval_in_model(extras['gp_lc_pred'],start) + 
                start['mean_lc'])

            lc=(pmx.eval_in_model(extras['model_lc'](t_lc_pred),start) -
                start['mean_lc'])

        xfold = fold(t_lc_pred, period, t0)
        inds = np.argsort(xfold)

        y_gp_pred = gp_pred 
        ylc = lc
        xfold = xfold[inds]
        foldedyvals = extras['y'][inds] - gp_pred[inds]
        print("generated model with new params")
        return (y_gp_pred,ylc, xfold, foldedyvals)


    def plot_new_gp_lines(event):

        mody, ylc, fmodx, fmody = load_new_lc_gps_param()

        gp_unfolded.set_ydata(mody)

        gp_folded.set_xdata(fmodx)
        gp_folded.set_ydata(fmody)

        fig.canvas.draw_idle()
        print("plotted GP models with new params")
    exec_btn.on_clicked(plot_new_gp_lines)
    plt.show()


interactive_pymc3_GP_plot_widget(x, y, lit_period, lit_t0)