import lightkurve as lk
import astropy.table as astab
import pandas as pd
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
from tqdm import tqdm
import pylab
pylab.rcParams['figure.figsize'] = (16, 8)
import warnings
import astropy.table as astab

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore",message='ERROR:lightkurve.search')
# warnings.filterwarnings('ignore', message=f'No data found for target')
print(astropy.__version__)

import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
import exoplanet as xo

import arviz as az
from corner import corner

from scipy.signal import savgol_filter


import numpy as np
dd = "/Users/kjaehnig/CCA_work/GAT/"


hq_jk_allstar_tess_edr3 = astab.Table.read(dd+'dr17_joker/unimodal_joker_sample_joined_w_tess_edr3.fits', format='fits')

calibverr = astab.Table.read(dd+'dr17_joker/allVisit-dr17-synspec-calib-verr.fits', format='fits', hdu=1)
print(calibverr.info)
sysapodat = astab.Table.read(dd+"dr17_joker/tic20215452_apogee_visit_data.fits", hdu=1, format='fits')
print(len(sysapodat))
sysapodat = astab.join(sysapodat, calibverr, keys=('VISIT_ID','VISIT_ID'))
print(len(sysapodat))



target_tic = 'TIC 20215452'

target_lk = lk.search_lightcurve(target_tic,
             mission='TESS',
            cadence='short',
            author='SPOC'
             )
print(target_lk)
unpro_lks = target_lk.download_all(quality_bitmask='hardest').stitch()
lks = target_lk.download_all(quality_bitmask='hardest').stitch(
        ).flatten(window_length=201)#.remove_outliers(sigma_lower=10**9, sigma_upper=.5)

jk_row = hq_jk_allstar_tess_edr3[hq_jk_allstar_tess_edr3['ID']==int(target_tic.split(' ')[1])][0]
periods = np.linspace(max(0.34,jk_row['MAP_P']-jk_row['MAP_P_err']), 
                    min(jk_row['MAP_P']+0.5*jk_row['MAP_P'],jk_row['MAP_P']+jk_row['MAP_P_err'])
                      ,1000)
joker_per = jk_row['MAP_P']
print(joker_per, jk_row['MAP_P_err'])



lks = lks.remove_nans()  #[lks.remove_nans()]
lks = lks.remove_outliers(sigma_lower=10**6, sigma_upper=.5)
star1_bls = lks.to_periodogram('bls',
                                period=np.exp(np.linspace(np.log(0.1), np.log(100),1000)),
                                frequency_factor = 1000, duration=np.linspace(0.005,0.09,100))
star1_per = star1_bls.period_at_max_power
star1_t0 = star1_bls.transit_time_at_max_power
star1_dur = star1_bls.duration_at_max_power
print(star1_per, star1_t0, star1_dur)
# lks.fold(period=star1_per, epoch_time=star1_t0).scatter()


lks_flat = unpro_lks.flatten(window_length=3)
# lks_flat.fold(normalize_phase=True, period=star1_per, epoch_time=star1_t0).scatter()



def interactive_lightcurve_fold_flattening_widget(lks, 
                                                  per0 = star1_per,
                                                  t0=star1_t0,
                                                  wl0=101
                                                 ):
    lks = lks.remove_nans()
    fig = plt.figure(figsize=(8,4),constrained_layout=False)
    ax_grid = GridSpec(nrows=100, ncols=100)
    ax_wl = fig.add_subplot(ax_grid[-5:,:60])
    ax_per = fig.add_subplot(ax_grid[-12:-7,:60])
    ax_t0 = fig.add_subplot(ax_grid[-19:-14, :60])
    ax_plt = fig.add_subplot(ax_grid[:-30,:])
    
    wl_btn = fig.add_subplot(ax_grid[-5:, 80:])
    per_btn = fig.add_subplot(ax_grid[-12:-7, 80:])
    t0_btn =  fig.add_subplot(ax_grid[-19:-14, 80:])


    ini_fold = lks.flatten(
    				window_length=wl0
    						).fold(
    							normalize_phase=True,
    							period=per0,
    							epoch_time=t0
    							)

    ini_plot, = ax_plt.plot(ini_fold.time.value,
    							ini_fold.flux.value,
    							rasterized=True,
    							marker='.',
    							markersize=5,
    							alpha=0.25,
    							ls='None')
    ymin = ini_fold.flux.min()
    ymax = ini_fold.flux.max()
    ax_plt.set_ylim(ymin - 0.1*ymin,
                    ymax + 0.1*ymax)

    wl_slider = Slider(
                ax = ax_wl,
                label = r"$L_{window}$",
                valmin = 3,
                valmax = 3*wl0,
                valinit = wl0,
                valfmt="%i",
                orientation='horizontal'
    )
    
    wl_button = TextBox(
                ax=wl_btn,
                label=r'$L_{w}$',
                initial=101
                )
    def update_wl(wlval):
        wl_val = int(wlval)
        if wl_val%2 == 0:
            wl_val += 1
        updated_lks = lks.flatten(window_length=wl_val)
        # ax_plt.clear()
        updated_lks = updated_lks.fold(
                normalize_phase=True, 
                period=float(per_slider.val), 
                epoch_time=t0_slider.val)

        ax_plt.set_xlim(updated_lks.time.value.min(),
                        updated_lks.time.value.max())
        ymin = updated_lks.flux.min()
        ymax = updated_lks.flux.max()

        ax_plt.set_ylim(ymin - 0.1*ymin,
                        ymax + 0.1*ymax)

        ini_plot.set_xdata(updated_lks.time.value)
        ini_plot.set_ydata(updated_lks.flux.value)
        param_str = f'WL: {wl_val}, P: {per_slider.val:3f}, t0: {t0_slider.val:2f}'
        ax_plt.set_title(param_str)
        fig.canvas.draw_idle()
        wl_slider.valinit = wl_val
        wl_slider.reset()
    wl_button.on_submit(update_wl)

    per_slider = Slider(
                ax = ax_per,
                label = r'$Period\ [d]$',
                valmin = per0/2.,
                valmax = 2*per0,
                valinit = per0,
                orientation='horizontal'
    )
    
    per_button = TextBox(
    			ax=per_btn,
    			label='P[d]',
                initial=np.round(per0,3)
    			)
    def update_period(perval):
        wl_val = int(wl_slider.val)
        if wl_val%2 == 0:
            wl_val += 1
        updated_lks = lks.flatten(window_length=wl_val)
        # ax_plt.clear()
        updated_lks = updated_lks.fold(
                normalize_phase=True, 
                period=float(perval), 
                epoch_time=t0_slider.val)

        ax_plt.set_xlim(updated_lks.time.value.min(),
        				updated_lks.time.value.max())
        ymin = updated_lks.flux.min()
        ymax = updated_lks.flux.max()

        ax_plt.set_ylim(ymin - 0.1*ymin,
						ymax + 0.1*ymax)

        ini_plot.set_xdata(updated_lks.time.value)
        ini_plot.set_ydata(updated_lks.flux.value)
        param_str = f'WL: {wl_val}, P: {per_slider.val:3f}, t0: {t0_slider.val:2f}'
        ax_plt.set_title(param_str)
        fig.canvas.draw_idle()
        per_slider.valinit = float(perval)
        per_slider.reset()
    per_button.on_submit(update_period)

    t0_slider = Slider(
                ax = ax_t0,
                label = r'$t_{0}$',
                valmin = min(lks.time.value),
                valmax = max(lks.time.value),
                valinit = t0,
                orientation = 'horizontal'
    )

    t0_button = TextBox(
                ax=t0_btn,
                label=r'$t_{0}$',
                initial=np.round(star1_t0.value,3)
                )
    def update_t0(t0val):
        wl_val = int(wl_slider.val)
        if wl_val%2 == 0:
            wl_val += 1
        updated_lks = lks.flatten(window_length=wl_val)
        # ax_plt.clear()
        updated_lks = updated_lks.fold(
                normalize_phase=True, 
                period=float(per_slider.val), 
                epoch_time=t0_slider.val)

        ax_plt.set_xlim(updated_lks.time.value.min(),
                        updated_lks.time.value.max())
        ymin = updated_lks.flux.min()
        ymax = updated_lks.flux.max()

        ax_plt.set_ylim(ymin - 0.1*ymin,
                        ymax + 0.1*ymax)

        ini_plot.set_xdata(updated_lks.time.value)
        ini_plot.set_ydata(updated_lks.flux.value)
        param_str = f'WL: {wl_val}, P: {per_slider.val:3f}, t0: {t0_slider.val:2f}'
        ax_plt.set_title(param_str)
        fig.canvas.draw_idle()
        t0_slider.valinit = t0val
        t0_slider.reset()
    t0_button.on_submit(update_t0)




    def update_lks_plot(val):
        wl_val = int(wl_slider.val)
        if wl_val%2 == 0:
            wl_val += 1
        updated_lks = lks.flatten(window_length=wl_val)
        # ax_plt.clear()
        updated_lks = updated_lks.fold(
                normalize_phase=True, 
                period=per_slider.val, 
                epoch_time=t0_slider.val)

        ax_plt.set_xlim(updated_lks.time.value.min(),
        				updated_lks.time.value.max())
        ymin = updated_lks.flux.min()
        ymax = updated_lks.flux.max()

        ax_plt.set_ylim(ymin - 0.1*ymin,
						ymax + 0.1*ymax)

        ini_plot.set_xdata(updated_lks.time.value)
        ini_plot.set_ydata(updated_lks.flux.value)
        param_str = f'WL: {wl_val}, P: {per_slider.val:3f}, t0: {t0_slider.val:2f}'
        ax_plt.set_title(param_str)
        fig.canvas.draw_idle()

    wl_slider.on_changed(update_lks_plot)
    per_slider.on_changed(update_lks_plot)
    t0_slider.on_changed(update_lks_plot)

    plt.show()


interactive_lightcurve_fold_flattening_widget(unpro_lks,
                                              per0=star1_per.value,
                                              t0=star1_t0.value,
                                              wl0=101)
