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





# TIC_TARGET = "TIC 28159019"
# COVARIANCE_USE_TYPE = 'isochrones'



#DD = "/Users/kjaehnig/CCA_work/GAT/"
DD = hf.load_system_specific_directory()

def load_construct_run_pymc3_model(
                                    TIC_TARGET=20215452, 
                                    COVARIANCE_USE_TYPE='diagonal',
                                    mult_factor=1,
                                    Ntune=1000, 
                                    Ndraw=500, 
                                    chains=4, 
                                    sparse_factor=5, 
                                    nsig=5):

    mf = mult_factor
   
    ##sys.stdout = open('file', 'w')
    print(f'starting model setup and run for TIC-{TIC_TARGET}')
    ##sys.stdout.close()
    TIC_TARGET = f'TIC {TIC_TARGET}'


    pymc3_model_dict = hf.load_precompiled_pymc3_model_data(DD, TIC_TARGET,
                                                        sparse_factor=sparse_factor)
    # pymc3_model_dict = load_all_data_for_pymc3_model(TIC_TARGET, 
    #     sparse_factor=sparse_factor, nsig=nsig)

    tic_dest, fig_dest = hf.check_for_system_directory_rusty_side(DD,TIC_TARGET, return_directories=True)

    texp = pymc3_model_dict['texp']
    x_rv, y_rv, yerr_rv = pymc3_model_dict['x_rv'], pymc3_model_dict['y_rv'], pymc3_model_dict['yerr_rv']
    x, y, yerr = pymc3_model_dict['x'], pymc3_model_dict['y'], pymc3_model_dict['yerr']
    lk_sigma = pymc3_model_dict['lk_sigma']

    lit_period, lit_t0, lit_tn = pymc3_model_dict['lit_period'], pymc3_model_dict['lit_t0'], pymc3_model_dict['lit_tn']
    Ntrans, ecosw_tv = pymc3_model_dict['Ntrans'], pymc3_model_dict['ecosw_tv']
    print('ecosw_tv: ', ecosw_tv)
    if abs(ecosw_tv) > 0.01:
        ecosw_tv = np.sign(ecosw_tv) * 0.01

    SUFFIX = f'individual_priors_{int(mf)}x_isochrones'

    LOGiso_M1 = pymc3_model_dict['isores']['logm1']
    LOGiso_R1 = pymc3_model_dict['isores']['logr1']
    LOGiso_q = pymc3_model_dict['isores']['logq']
    LOGiso_k = pymc3_model_dict['isores']['logk']
    LOGiso_s = pymc3_model_dict['isores']['logs']


    trv = np.linspace(x_rv.min(), x_rv.max(), 5000)
    tlc = np.linspace(x.min(), x.max(), 5000)

    # rvK = xo.estimate_semi_amplitude(bls_period, x_rv, y_rv*u.km/u.s, yerr_rv*u.km/u.s, t0s=bls_t0)[0]
    # print(rvK)

    # mask = x < 400
    def build_model(mask=None, start=None, suffix=None, pymc3_model_dict=None):
        if mask is None:
            mask = np.ones(len(x), dtype='bool')
        with pm.Model() as model:

            # Systemic parameters
            mean_lc = pm.Normal("mean_lc", mu=0.0, sd=10.0)
            mean_rv = pm.Normal("mean_rv", mu=0.0, sd=50.0)

            u1 = xo.QuadLimbDark("u1")
            u2 = xo.QuadLimbDark("u2")

            # Parameters describing the primary
            log_M1 = pm.Normal("log_M1", mu=LOGiso_M1[0], sigma=mf*LOGiso_M1[1], testval=LOGiso_M1[0])
            log_R1 = pm.Normal("log_R1", mu=LOGiso_R1[0], sigma=mf*LOGiso_R1[1], testval=LOGiso_R1[0])

            M1 = pm.Deterministic("M1", tt.exp(log_M1))
            R1 = pm.Deterministic("R1", tt.exp(log_R1))

            
            # Secondary ratios
            log_k = pm.Normal("log_k", mu=LOGiso_k[0], sigma=mf*LOGiso_k[1], testval = LOGiso_k[0])  # radius ratio        
            log_q = pm.Normal("log_q", mu=LOGiso_q[0], sigma=mf*LOGiso_q[1], testval = LOGiso_q[0])  # mass ratio
            log_s = pm.Normal("log_s", mu=LOGiso_s[0], sigma=mf*LOGiso_s[1], testval = LOGiso_s[0])  # surface brightness ratio
      
            
            k = pm.Deterministic("k", tt.exp(log_k))
            q = pm.Deterministic("q", tt.exp(log_q))
            s = pm.Deterministic("s", tt.exp(log_s))


            # Parameters describing the orbit
            b = xo.ImpactParameter("b", ror=k, testval=0.5)

    #         log_period = pm.Normal("log_period", mu=np.log(lit_period), sigma=5.0)
    #         period = pm.Deterministic("period", tt.exp(log_period))
            t0 = pm.Normal("t0", mu=lit_t0, sigma=1.0)
            tn = pm.Normal("tn", mu=lit_tn, sigma=1.0)
            period = pm.Deterministic("period", (tn - t0) / Ntrans)
            # Parameters describing the eccentricity: ecs = [e * cos(w), e * sin(w)]
    #         ecosw_tv=0.01
            sqrt_ecosw = np.sign(ecosw_tv) * np.sqrt(abs(ecosw_tv))
            # ecs is now sqrt(ecs) even if variable name is still ecs
            ecs = pmx.UnitDisk("ecs", testval=np.array([sqrt_ecosw, 0.0]))
            # remove sqrt from ecc, rewrite as ecosW and esinW
            
            
            
            ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2))
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
            
            
            
            sigma_lc = pm.InverseGamma(
                "sigma_lc",
                testval= np.mean(yerr),
                **pmx.estimate_inverse_gamma_parameters(0.1,5.0)
            )
            
            
            sigma_gp = pm.InverseGamma(
                "sigma_gp",
                testval= lk_sigma,
                **pmx.estimate_inverse_gamma_parameters(0.1,10.0),
            )
            rho_gp = pm.InverseGamma(
                "rho_gp",
                testval= 2.0 * lit_period,
                **pmx.estimate_inverse_gamma_parameters(0.1,10.0)
            )
    #         sigma_lc = np.mean(yerr)
    #         sigma_gp = lk_sigma
    #         rho_gp = 0.25*lit_period
            print(sigma_lc, sigma_gp, rho_gp)
            kernel_lc = terms.SHOTerm(sigma=sigma_gp, rho=rho_gp, Q=1.0 / 3.)

    #         # Noise model for the radial velocities
    #         sigma_rv = pm.InverseGamma(
    #             "sigma_rv",
    #             testval=1.0,
    #             **pmx.estimate_inverse_gamma_parameters(0.1, 15.0)
    #         )
    #         sigma_rv_gp = pm.InverseGamma(
    #             "sigma_rv_gp",
    #             testval=1.0,
    #             **pmx.estimate_inverse_gamma_parameters(0.1, 15.0)
    #         )
    #         rho_rv_gp = pm.InverseGamma(
    #             "rho_rv_gp",
    #             testval=2.0,
    #             **pmx.estimate_inverse_gamma_parameters(0.1, 25.0)
    #         )
    #         kernel_rv = terms.SHOTerm(sigma=sigma_rv_gp, w0=rho_rv_gp, Q=1.0 / 3.)

            # Set up the light curve model
            lc = xo.SecondaryEclipseLightCurve(u1, u2, tt.exp(log_s))

            def model_lc(t):
                return (
                    mean_lc
                    + 1e3
                    * lc.get_light_curve(orbit=orbit, r=R2, t=t, texp=texp)[:,0]
                )

    #         pm.Deterministic(
    #             "lc_pred",
    #             model_lc(x)
    #         )
            
            # Condition the light curve model on the data
            gp_lc = GaussianProcess(kernel_lc, t=x[mask], yerr=sigma_lc)
            gp_lc.marginal("obs_lc", observed=y[mask] - model_lc(x[mask]))


    #         # Set up the radial velocity model

            log_sigma_rv =  pm.Bound(pm.Normal, upper=np.log(10))(
                "log_sigma_rv", mu=np.log(np.median(yerr_rv)), sd=10.0, testval=np.log(np.median(yerr_rv))
            )

            def model_rv(t):
                return orbit.get_radial_velocity(t, output_units=u.km/u.s) + mean_rv
                
            rv_model = model_rv(x_rv)
            
    #         def model_K(t, period, t0):
    #             rvs = model_rv(t)
    #             modK = xo.estimate_semi_amplitude(period, t, rvs, yerr_rv, t0).to(u.km/u.s)
    #             return modK
            
            err = tt.sqrt(yerr_rv**2. + tt.exp(2*log_sigma_rv))
            
            pm.Normal("obs",mu=rv_model, sd=err, observed=y_rv)

            ## compute phased RV signal
    #         n = 2.*np.pi * (1./period)
    #         phi = (t0 * n) - omega
    #         phase = np.linspace(0, 1, 500)
    #         M_pred = 2 * np.pi * phase - (phi + omega)
    #         f_pred = xo.orbits.get_true_anomaly(M_pred, ecc + tt.zeros_like(M_pred))
            
    # #         K = xo.estimate_semi_amplitude(period, t, rv_model, yerr_rv, t0).to(u.km/u.s)
    #         K = (tt.max(rv_model) - tt.min(rv_model)) / 2.
        
    #         rvphase = pm.Deterministic(
    #             "rvphase", K * (tt.cos(omega) * (tt.cos(f_pred) + ecc) - tt.sin(omega) * tt.sin(f_pred))
    #         )
    #         n = 2.*np.pi * (1./period)
    #         phi = (t0 * n) - omega
    #         phase = np.linspace(0, 1, 500)
    #         M_pred = 2 * np.pi * phase - (phi + omega)
    #         f_pred = xo.orbits.get_true_anomaly(M_pred, ecc + tt.zeros_like(M_pred))
            
    # #         K = xo.estimate_semi_amplitude(period, t, rv_model, yerr_rv, t0).to(u.km/u.s)
    #         K = (tt.max(rv_model) - tt.min(rv_model)) / 2.
        
    #         rvphase = pm.Deterministic(
    #             "rvphase", K * (tt.cos(omega) * (tt.cos(f_pred) + ecc) - tt.sin(omega) * tt.sin(f_pred))
    #         )
            
            # Optimize the logp
            if start is None:
                start = model.test_point

                
            extras = dict(
                x=x[mask],
                y=y[mask],
                x_rv = x_rv,
                y_rv = y_rv,
                yerr_rv = yerr_rv,
                model_lc=model_lc,
                model_rv=model_rv,
                gp_lc_pred=gp_lc.predict(y[mask] - model_lc(x[mask])),
            )
            
            
            # First the RV parameters
            print(model.check_test_point())
            opti_logp = []
            filename_list = []
            map_vars_dict = {
                't0_tn' : [tn,t0],
                'u1_u2' : [u1, u2],
                'log_q' : log_q,
                'log_s' : log_s,
                'log_M1': log_M1,
                'log_R1': log_R1,
                'mean_lc': mean_lc,
                'mean_rv': mean_rv,
                'lsig_rv':log_sigma_rv
            }
            list_of_map_vars = list(map_vars_dict.keys())
            

            filename_base = f"{fig_dest}{TIC_TARGET.replace(' ','_')}_{suffix}"
            
            plot = hf.plot_MAP_rv_curve_diagnostic_plot(model, start, extras, mask, 
                                                     title=' after start point opt step',
                                                     filename=filename_base + ' after start point opt step'.replace(' ','_'),
                                                     RETURN_FILENAME=True, pymc3_model_dict=pymc3_model_dict)
            filename_list.append(plot)
            
            map_soln, info_ = pmx.optimize(start, log_k, return_info=True)
            plot = hf.plot_MAP_rv_curve_diagnostic_plot(model, map_soln, extras, mask, 
                                                     title='after log_k opt step',
                                                     filename=filename_base + ' after log_k opt step.png'.replace(' ','_'),
                                                     RETURN_FILENAME=True, pymc3_model_dict=pymc3_model_dict)
            filename_list.append(plot)
            
            map_soln, info_ = pmx.optimize(map_soln, b, return_info=True)
            plot = hf.plot_MAP_rv_curve_diagnostic_plot(model, map_soln, extras, mask, 
                                                     title=' after b opt step',
                                                     filename = filename_base + ' after b opt step'.replace(' ','_'),
                                                     RETURN_FILENAME=True, pymc3_model_dict=pymc3_model_dict)
            filename_list.append(plot)




            map_soln, info_ = pmx.optimize(map_soln, ecs, return_info=True)
            plot = hf.plot_MAP_rv_curve_diagnostic_plot(model, map_soln, extras, mask, 
                                                     title='model after [ecs] opt step',
                                                     filename=filename_base + ' model after [ecs] opt step'.replace(' ','_'),
                                                     RETURN_FILENAME=True, pymc3_model_dict=pymc3_model_dict)
            ecs_logp = -info_['fun']
            filename_list.append(plot)
            if ~np.isfinite(-info_['fun']):
                list_of_map_vars.append('ecs')
                map_vars_dict['ecs'] = ecs
            
            

            best_map_list = []
            whileloop_failsafe = 0
            while len(list_of_map_vars) > 0:
                highest_logp = -np.inf
                try:
                    next_map_var = []
                    next_map_soln = []
                    next_opti_logp = np.nan
                    for ii in list_of_map_vars:
    #                     print(f"optimizing {ii}") 
                        map_soln_ii, info_ = pmx.optimize(map_soln, map_vars_dict[ii], return_info=True, verbose=False)
                        
                        if np.isfinite(-info_['fun']):
                            if -info_['fun'] > highest_logp:
                                next_map_var = ii
                                next_map_soln = map_soln_ii
                                next_opti_logp = -info_['fun']
                                highest_logp = -info_['fun']
                                
                    opti_logp.append(next_opti_logp)
                    map_soln = next_map_soln
                    list_of_map_vars.remove(next_map_var)
                    best_map_list.append(next_map_var)
                    
                    print("-"*50)
                    print(f"N_best: {len(best_map_list)}, N_left: {len(list_of_map_vars)}, {next_map_var}")
                    plotn = hf.plot_MAP_rv_curve_diagnostic_plot(model, map_soln, extras, mask, title=f' after {next_map_var} opt step', 
                                                              filename=filename_base + f' after {next_map_var} opt step'.replace(' ','_'),
                                                              RETURN_FILENAME=True, pymc3_model_dict=pymc3_model_dict)
                    filename_list.append(plotn)

                except:
                    print("loop failed")
                    whileloop_failsafe += 1
                    print(whileloop_failsafe)
                if whileloop_failsafe > 20:
                    break
            
            
            map_soln, info_ = pmx.optimize(map_soln, [sigma_lc, sigma_gp, rho_gp], return_info=True)
            plot = hf.plot_MAP_rv_curve_diagnostic_plot(model, map_soln, extras, mask, 
                                                     title=' after GP params opt step',
                                                     filename=filename_base + ' after GP params opt step'.replace(' ','_'),
                                                     RETURN_FILENAME=True, pymc3_model_dict=pymc3_model_dict)
            filename_list.append(plot)
            
            map_soln, info_ = pmx.optimize(map_soln, 
                                           return_info=True)
            plot = hf.plot_MAP_rv_curve_diagnostic_plot(model, map_soln, extras, mask, 
                                                     title=' after final opt step',
                                                     filename=filename_base+' after final opt step'.replace(' ','_'),
                                                     RETURN_FILENAME=True, pymc3_model_dict=pymc3_model_dict)
            filename_list.append(plot) 






        return model, map_soln, extras, start, opti_logp, filename_list



    model, map_soln, extras, start, opti_logp, filename_list = \
        build_model(suffix=SUFFIX, pymc3_model_dict=pymc3_model_dict)

    import imageio
    images = []

    filename_list.append(filename_list[-1])
    filename_list.append(filename_list[-1])
    filename_list.append(filename_list[-1])
    filename_list.append(filename_list[-1])
    for filename in filename_list:
        images.append(imageio.imread(filename))
    imageio.mimsave(tic_dest+f"/{TIC_TARGET.replace(' ','_')}_{SUFFIX}__diagnostic_movie_test.gif", images, fps=0.75)

    print("#" * 50)
    print("#"*19 +"  FINISHED  " + "#"*19)
    print("#"*50)


    with model:
        mod = pmx.eval_in_model(
            extras['model_lc'](extras['x']) + extras['gp_lc_pred'],
            map_soln,
        )
        
    resid = y - mod
    rms = np.sqrt(np.median(resid ** 2))
    mask = np.abs(resid) < 5 * rms

    plt.figure(figsize=(10, 5))
    plt.plot(x, resid, "k", label="data")
    plt.plot(x[~mask], resid[~mask], "xr", label="outliers")
    plt.axhline(0, color="#aaaaaa", lw=1)
    plt.ylabel("residuals [ppt]")
    plt.xlabel("time [days]")
    plt.legend(fontsize=12, loc=3)
    _ = plt.xlim(x.min(), x.max())
    plt.savefig(fig_dest + f"{TIC_TARGET}_{SUFFIX}_sigma_cliped_lightcurve_plot.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("#" * 50)
    print("Starting 2nd round of MAP optimizations.")
    print("#" * 50)
    SUFFIX = SUFFIX + "_2nd_rnd"
    model, map_soln, extras, start, opti_logp,_ = build_model(
            mask, map_soln, suffix=SUFFIX, pymc3_model_dict=None)

    # ###### quick fix to save x and y to the main file dump #######
    # file = open(f"/Users/kjaehnig/CCA_work/GAT/pymc3_models/{TIC_TARGET}_pymc3_Nt{Ntune}_Nd{Ndraw}_Nc{chains}_{SUFFIX}.pickle",'rb')
    # indres = pk.load(file)
    # file.close()

    # indres['lcdat'] = {'x':x[mask],'y':y[mask],'yerr':yerr[mask]}
    # indres['rvdat'] = {'x_rv':x_rv,'y_rv':y_rv,'yerr_rv':yerr_rv}
    # with model:
    #     gp_pred = (
    #         pmx.eval_in_model(extras["gp_lc_pred"], map_soln)
    #     )
    # indres['gp_pred'] = gp_pred
    # file = open(f"/Users/kjaehnig/CCA_work/GAT/pymc3_models/{TIC_TARGET}_pymc3_Nt{Ntune}_Nd{Ndraw}_Nc{chains}_{SUFFIX}.pickle",'wb')
    # pk.dump(indres, file)
    # file.close()
    # print("DONE")
    # return
    ########
    # Ntune = 1000
    # Ndraw = 500
    # chains = 4

    random_seeds = [int(f'26113668{ii+1}') for ii in range(chains)]
    print(random_seeds)
    with model:
        trace = pm.sample(
            tune=Ntune,
            draws=Ndraw,
            start=map_soln,
            # Parallel sampling runs poorly or crashes on macos
            cores=chains,
            chains=chains,
            target_accept=0.99,
            return_inferencedata=True,
            random_seed=random_seeds,##[261136681, 261136682,261136683,261136684],#261136685, 261136686,261136687,261136688],
            init='jitter+adapt_full'
        )


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

    flat_samps = trace.posterior.stack(sample=('chain', 'draw')) #trace.posterior.stack(sample=("chain", "draw"))

    rvvals = compute_value_in_post(model, trace, extras['model_rv'](trv), size=512)
    lcvals = compute_value_in_post(model, trace, extras['model_lc'](tlc), size=512)

    rvact = compute_value_in_post(model, trace, extras['model_rv'](x_rv), size=512)
    lcact = compute_value_in_post(model, trace, extras['model_lc'](x), size=512)

    # print(map_soln)
    with model:
        gp_pred = (
            pmx.eval_in_model(extras["gp_lc_pred"], map_soln)
        )

    if 'pymc3_models' not in os.listdir(DD):
        os.mkdir(DD+'pymc3_models')
    pymc3_DD = DD + 'pymc3_models'
    file = open(f"{pymc3_DD}/{TIC_TARGET}_pymc3_Nt{Ntune}_Nd{Ndraw}_Nc{chains}_{SUFFIX}.pickle",'wb')
    pk.dump({
            'trace':trace,
             'mask':mask,
            'map_soln':map_soln,
            'model':model,
            'trv':trv,
            'tlc':tlc,
            'lcvals': lcvals,
            'rvvals': rvvals,
            'lcact': lcact,
            'rvact': rvact,
            'gp_pred': gp_pred,
            'lcdat': {'x':x[mask],'y':y[mask],'yerr':yerr[mask]},
            'rvdat': {'x_rv':x_rv,'y_rv':y_rv,'yerr_rv':yerr_rv}
            },
            file)
    file.close()





    p_med = flat_samps['period'].median().values
    t0_med = flat_samps['t0'].median().values
    mean_rv = flat_samps['mean_rv'].median().values
    mean_lc = flat_samps['mean_lc'].median().values
    # gp_pred = flat_samps['gp_pred'].median().values

    fig, axes = plt.subplots(figsize=(10,10), ncols=1, nrows=2)
    # print(flat_samps['ecc'].median())

    axes[0].errorbar(hf.fold(x_rv, p_med, t0_med),
                  y_rv, yerr=yerr_rv, fmt=".k")
    # rvvals = indres['rvvals']
    # lcvals = indres['lcvals']
        
    t_fold = hf.fold(trv, p_med, t0_med)
    inds = np.argsort(t_fold)
    pred = np.percentile(rvvals, [16, 50, 84], axis=0)
    axes[0].plot(t_fold[inds], pred[1][inds], color='C1', zorder=2)

    pred = np.percentile(rvvals, [16, 84], axis=0)
    art = axes[0].fill_between(t_fold[inds], pred[0][inds], pred[1][inds], color='C1', alpha=0.5, zorder=1)
    art.set_edgecolor("none")

    pred = np.percentile(rvvals, [5, 95], axis=0)
    art = axes[0].fill_between(t_fold[inds], pred[0][inds], pred[1][inds], color='C1', alpha=0.25, zorder=0)
    art.set_edgecolor("none")

    # pred = np.percentile(rvvals, [1, 99], axis=0)
    # art = axes[0].fill_between(t_fold, pred[0], pred[1], color='C1', alpha=0.10, zorder=0)
    # art.set_edgecolor("none")
    # axes[0].set_ylim(-40, 40)
    # axes[1].set_ylim(-40, 40)
    axes[0].set_ylabel("RV [kms]")

    x,y = extras['x'],extras['y']
    # with model:
    #     gp_pred = (
    #         pmx.eval_in_model(extras["gp_lc_pred"], post_map_soln)
    #     )

        
    axes[1].errorbar(hf.fold(x, p_med, t0_med),
                  y-gp_pred, fmt=".k", ms=1, zorder=-1)

    t_fold = hf.fold(tlc, p_med, t0_med)
    inds = np.argsort(t_fold)
    pred = np.percentile(lcvals, [16, 50, 84], axis=0)
    axes[1].plot(t_fold[inds], pred[1][inds], color='C1', zorder=2)

    pred = np.percentile(lcvals, [16, 84], axis=0)
    art = axes[1].fill_between(t_fold[inds], pred[0][inds], pred[1][inds], color='C1', alpha=0.5, zorder=1)
    art.set_edgecolor("none")

    pred = np.percentile(lcvals, [5, 95], axis=0)
    art = axes[1].fill_between(t_fold[inds], pred[0][inds], pred[1][inds], color='C1', alpha=0.25, zorder=0)
    art.set_edgecolor("none")

    # pred = np.percentile(flat_samps['lc_pred'][inds], [1, 99], axis=-1)
    # art = axes[1].fill_between(t_fold[inds], pred[0], pred[1], color='C1', alpha=0.10, zorder=0)
    # art.set_edgecolor("none")

    axes[1].set_xlabel("phase [days]")
    axes[1].set_ylabel("flux [ppt]")

    plt.savefig(fig_dest + f"{TIC_TARGET.replace(' ','_').replace('-','_')}_{SUFFIX}_rvphase_plot.png",dpi=150,bbox_inches='tight')
    plt.close()

    ##sys.stdout.close()


# TIC_TARGET='20215452', 
# COVARIANCE_USE_TYPE='diagonal',
# mult_factor=1,
# Ntune=1000, 
# Ndraw=500, 
# chains=4, 
# sparse_factor=5, 
# nsig=5):

result = OptionParser()
result.add_option('--tic', dest='TIC_TARGET', default=20215452, type='int', 
                help='tic ID number of target (defaults to 20215451)')
result.add_option('--CovType', dest='COVARIANCE_USE_TYPE', default='diagonal', type='str',
                help='type of covariance matrix to use in multivariate prior (defaults to "diagonal")')
result.add_option("--mf", dest='mult_factor', default=1, type='int',
                help='multiplicative factor by which to increase multivariate prior variances (default: 1)')
result.add_option("--nt", dest="Ntune", default=1000, type='int',
                help="number of tuning draws to perform during sampling (default: 1000)")
result.add_option("--nd", dest="Ndraw", default=500, type='int',
                help="number of sample draws to perform during sampling (default: 500)")
result.add_option("-c", dest='chains', default=2, type='int',
                help='number of chains to run during sampling (default: 2)')
result.add_option("--sf", dest='sparse_factor', default=5, type='int',
                help='how sparse to make the lightcurve data before running pymc3 (default: 5)')
result.add_option("--nsig", dest='nsig', default=5, type='int',
                help='number of sigma to consider in constructing isochrones BinMod distributions (default: 5)')


if __name__ == "__main__":
    opt,arguments = result.parse_args()
    load_construct_run_pymc3_model(**opt.__dict__)

# tic_systems_of_interest = [
#     28159019,
#     272074664,
#     20215452,
#     99254945,
#     144441148,
#     169820068,
#     126232983,
#     164458426,
#     164527723,
#     165453878,
#     258108067,
#     271548206,
#     365204192
#     ]

# mult_factor = 1

# for TICs in tic_systems_of_interest:
#     for model_type in ['diagonalized']:#,'diagonalized4']:
#         if model_type == 'diagonalized4':
#             mult_factor = int(model_type[-1])
#             model_type = 'diagonalized'
#         load_construct_run_pymc3_model(TIC_TARGET=f'TIC {TICs}',
#             COVARIANCE_USE_TYPE=model_type, mult_factor=mult_factor,
#             sparse_factor=2, 
#             Ndraw=500, Ntune=1000, chains=6)
