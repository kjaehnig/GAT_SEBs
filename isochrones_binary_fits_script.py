import astropy.table as astab
from isochrones import BinaryStarModel,get_ichrone
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from zero_point import zpt
from corner import corner
import arviz as az
import helper_functions as hf
from optparse import OptionParser
import itertools
import os

DD = hf.load_system_specific_directory()

import sys
what_machine_am_i_on = sys.platform

if what_machine_am_i_on == 'darwin':
    hq_joker_edr3_apogee_tess_df = astab.Table.read(DD+"dr17_joker/unimodal_joker_sample_joined_w_tess_edr3_REDUX.fits").to_pandas()

if what_machine_am_i_on == 'linux' or what_machine_am_i_on == 'linux2':
    hq_joker_edr3_apogee_tess_df = astab.Table.read(DD+"unimodal_joker_sample_joined_w_tess_edr3_REDUX.fits").to_pandas()


# hq_joker_edr3_apogee_tess_df = astab.Table.read(DD+"unimodal_joker_sample_joined_w_tess_edr3_REDUX.fits").to_pandas()

### PILFERED AND MODIFIED FROM MARINA KOUNKEL'S GITHUB REPOSITORY FOR THE AURIGA NEURAL NET
print("starting...")

zpt.load_tables()

def get_edr3_dr2_xmatch(data):
    """ Main code format taken from Auriga Neural Net github page. Takes a
    dataframe of Gaia DR2 source_ids and cross-matches them with the cross
    matching done by the Gaia collaboration between the DR2 catalog and the
    EDR3 catalog.

    Parameters
    ----------
        data : pandas dataframe
            Stars for which you want to download data for. Must include the
            columns 'Cluster', 'source_id', and 'tag'
    Returns
    -------
        dat : pandas dataframe
            Results from Gaia ADQL query converted to pandas dataframe.
    """
    from astroquery.gaia import Gaia 
    from astropy.table import Table

    Gaia.login(user='kjaehnig',
                password='Legacyofash117!', verbose=True)
    table = Table([data['GAIAEDR3_SOURCE_ID'], data['ID']],# data['bss_flag']],
                names=['source_id','tic'])
    # orig_id = str(datrow['source_id'].squeeze())
    # print(orig_id)
    # table = Table([datrow['source_id']],names=['orig_id'])
    res = Gaia.launch_job_async(query="select tbl.tic, \
            gedr3.source_id, gedr3.parallax, gedr3.parallax_error, gedr3.phot_g_mean_mag, \
            gedr3.phot_bp_mean_mag, gedr3.phot_rp_mean_mag, gedr3.nu_eff_used_in_astrometry, \
            gedr3.phot_g_mean_flux, gedr3.phot_g_mean_flux_error, \
            gedr3.phot_bp_mean_flux, gedr3.phot_bp_mean_flux_error, \
            gedr3.phot_rp_mean_flux, gedr3.phot_rp_mean_flux_error, \
            gedr3.pseudocolour, gedr3.ecl_lat, gedr3.astrometric_params_solved, \
            dr3_dist.r_med_geo, dr3_dist.r_lo_geo, dr3_dist.r_hi_geo, \
            dr3_dist.r_med_photogeo, dr3_dist.r_lo_photogeo, dr3_dist.r_hi_photogeo \
            from gaiaedr3.gaia_source as gedr3 \
            inner join TAP_UPLOAD.mytbl as tbl \
                on gedr3.source_id = tbl.source_id \
            inner join external.gaiaedr3_distance as dr3_dist \
                on gedr3.source_id = dr3_dist.source_id",
            upload_resource=table, upload_table_name='mytbl')
    dat = res.get_results().to_pandas() 
    # dat['Cluster'] = dat.Cluster.str.decode("UTF-8")
    return dat




def get_2mass_mag_uncertainties(data):
    from astroquery.gaia import Gaia
    table=astab.Table([data['source_id']],names=['source_id'])
    j = Gaia.launch_job_async(query="select g.source_id,tm.j_msigcom as ej,tm.h_msigcom as eh,tm.ks_msigcom as ek \
        FROM gaiadr2.gaia_source AS g \
        inner join TAP_UPLOAD.table_test AS tc \
        ON g.source_id = tc.source_id \
        LEFT OUTER JOIN gaiadr2.tmass_best_neighbour AS xmatch \
        ON g.source_id = xmatch.source_id \
        LEFT OUTER JOIN gaiadr1.tmass_original_valid AS tm \
        ON tm.tmass_oid = xmatch.tmass_oid", upload_resource=table, upload_table_name="table_test")
    d = j.get_results().to_pandas()
    return d


def generate_params_for_multinest(ticnum, return_edr3=False):
    
    ticrow = hq_joker_edr3_apogee_tess_df.loc[hq_joker_edr3_apogee_tess_df['ID'] == ticnum]
    print(ticrow.columns.values)
    tmass_errs = get_2mass_mag_uncertainties(ticrow)
    if tmass_errs.shape[0] > 0:
        assert tmass_errs.source_id.squeeze() == ticrow['source_id'].squeeze()
        eh,ej,ek = tmass_errs.eh.squeeze(), tmass_errs.ej.squeeze(), tmass_errs.ek.squeeze()
    if tmass_errs.shape[0] == 0:
        print("no tmass errors available.")
        eh,ej,ek = 0.02,0.02,0.02


    g = ticrow.phot_g_mean_mag.squeeze()
    bp = ticrow.phot_bp_mean_mag.squeeze()
    rp = ticrow.phot_rp_mean_mag.squeeze()
    eg =  (-2.5*np.log10(ticrow.phot_g_mean_flux.squeeze() -ticrow.phot_g_mean_flux_error.squeeze()) 
          + 2.5*np.log10(ticrow.phot_g_mean_flux.squeeze() +ticrow.phot_g_mean_flux_error.squeeze())) / 2.
    ebp = (-2.5*np.log10(ticrow.phot_bp_mean_flux.squeeze()-ticrow.phot_bp_mean_flux_error.squeeze()) 
          + 2.5*np.log10(ticrow.phot_bp_mean_flux.squeeze()+ticrow.phot_bp_mean_flux_error.squeeze())) / 2.
    erp = (-2.5*np.log10(ticrow.phot_rp_mean_flux.squeeze()-ticrow.phot_rp_mean_flux_error.squeeze()) 
          + 2.5*np.log10(ticrow.phot_rp_mean_flux.squeeze()+ticrow.phot_rp_mean_flux_error.squeeze())) / 2.
#     print(eg, ebp, erp)

    edr3_zpt = get_edr3_dr2_xmatch(ticrow)
    zpt_corr = zpt.get_zpt(edr3_zpt["phot_g_mean_mag"], edr3_zpt["nu_eff_used_in_astrometry"],
                   edr3_zpt["pseudocolour"],edr3_zpt["ecl_lat"],
                   edr3_zpt["astrometric_params_solved"],
                   _warnings=False)
    print(edr3_zpt)
    if len(zpt_corr) > 0:
        print("applying zpt correction")
        plx_corr = edr3_zpt.parallax.squeeze()  - zpt_corr[0]
    else:
        plx_corr = edr3_zpt.parallax.squeeze()

    eg3 = edr3_zpt.phot_g_mean_mag.squeeze()
    ebp3 = edr3_zpt.phot_bp_mean_mag.squeeze()
    erp3 = edr3_zpt.phot_rp_mean_mag.squeeze()
    eg3e =  (-2.5*np.log10(edr3_zpt.phot_g_mean_flux.squeeze() -edr3_zpt.phot_g_mean_flux_error.squeeze()) 
          + 2.5*np.log10(edr3_zpt.phot_g_mean_flux.squeeze() +edr3_zpt.phot_g_mean_flux_error.squeeze())) / 2.
    ebp3e = (-2.5*np.log10(edr3_zpt.phot_bp_mean_flux.squeeze()-edr3_zpt.phot_bp_mean_flux_error.squeeze()) 
          + 2.5*np.log10(edr3_zpt.phot_bp_mean_flux.squeeze()+edr3_zpt.phot_bp_mean_flux_error.squeeze())) / 2.
    erp3e = (-2.5*np.log10(edr3_zpt.phot_rp_mean_flux.squeeze()-edr3_zpt.phot_rp_mean_flux_error.squeeze()) 
          + 2.5*np.log10(edr3_zpt.phot_rp_mean_flux.squeeze()+edr3_zpt.phot_rp_mean_flux_error.squeeze())) / 2.

    params = {
#         'Teff':(ticrow.TEFF.squeeze(),ticrow.TEFF_ERR.squeeze()),
#         'logg':(ticrow.LOGG.squeeze(),ticrow.LOGG_ERR.squeeze()),
#         'feh':(ticrow.FE_H.squeeze(), ticrow.FE_H_ERR.squeeze()),
        'H':(ticrow.H.squeeze(), eh),
        'J':(ticrow.J.squeeze(), ej),
        'K':(ticrow.K.squeeze(), ek),
        'BP':(bp, ebp),
        'RP':(rp, erp),
        'G':(g, eg),
        'eBP3':(ebp3, ebp3e),
        'eRP3':(erp3, erp3e),
        'eG3':(eg3, eg3e),
        # 'parallax':(plx_corr , edr3_zpt.parallax_error.squeeze()),
    }

    if return_edr3:   
        return (params, edr3_zpt)
    else:
        return params


def add_tess_mag_to_params_dict(params):
    import ticgen

    mag_dict = {'Jmag':params['J'][0],
                'Hmag':params['H'][0],
                'Ksmag':params['K'][0],
                'Gmag':params['G'][0],
               'integration':2}

    mag_combos = list(itertools.combinations(list(mag_dict.keys())[:-1],2))

    mag_list = []
    for combo in mag_combos:
        mag_dict_samp = {
            combo[0]: mag_dict[combo[0]],
            combo[1]: mag_dict[combo[1]],
            'integration': 2.0
        }
        try:
            mag, _ = ticgen.calc_star(mag_dict_samp)
            mag_list.append(mag)
        except:
            continue

    print(f"there are {len(mag_list)} different mag values")
    print(np.std(mag_list))
    params['TESS'] = (np.median(mag_list), 0.05)

    return params


def calculate_only_tmag_w_G_J(params):
    g,ge = params['G']
    j,je = params['J']

    g_j = g - j
    g_j_e = np.sqrt(je**2. + ge**2.)

    T = (g + 0.00106 * g_j**3. + 0.01278 * g_j**2. - 0.46022 * g_j + 0.0211)

    t2 = abs( (0.00106 * g_j**3. * 3 * g_j_e) / g_j)

    t3 = abs( (0.01278 * g_j**2. * 2 * g_j_e) / g_j)

    t4 = abs( (0.46022 * g_j * 1 * g_j_e) / g_j)

    Te = np.sqrt( 0.015**2. + ge**2. + t2**2. + t3**2. + t4**2. )

    print(f"tmag, err : {T}, {Te}")

    return (T, Te)


def initialize_multinest_binary_model(ticnum, TestOnly=False):
    from isochrones.priors import GaussianPrior
    params, edr3_df = generate_params_for_multinest(ticnum, return_edr3=True)

    other_params = add_tess_mag_to_params_dict(params)

    params['TESS'] = calculate_only_tmag_w_G_J(params)

    print("$"*120)
    print(ticnum, calculate_only_tmag_w_G_J(params), other_params['TESS'])
    print("$"*120)
    # params = add_tess_mag_to_params_dict(params)

    mist = get_ichrone('mist', bands=['J','H','K','eBP3','eRP3','eG3','TESS'])
    
    if TestOnly == False:
        test = ''
    else:
        test = 't'
    binarymodel = BinaryStarModel(mist, **params, name=f'TIC_{ticnum}{test}')
    binarymodel.mnest_basename = DD+'ceph'+binarymodel.mnest_basename[1:]
    
    # distance = 1000./params['parallax'][0]
#     feh_bounds = (params['feh'][0]-3*params['feh'][1], params['feh'][0]+3*params['feh'][1])
    # print(feh_bounds)
    # plus_dist = (1000./(params['parallax'][0] + params['parallax'][1]))
    # mins_dist = (1000./(params['parallax'][0] - params['parallax'][1]))
#     print(plus_dist, mins_dist)

    # mdist_diff = distance - min(plus_dist, mins_dist)
    # pdist_diff = max(plus_dist, mins_dist) - distance
    
    # edist_estimate = (mdist_diff + pdist_diff) / 2.
    
    r_lo_pg, r_med_pg, r_hi_pg = edr3_df[['r_lo_photogeo','r_med_photogeo','r_hi_photogeo']].squeeze()

    distance = r_med_pg
    edist = .5 * ((distance - r_lo_pg) + (r_hi_pg - distance))

    binarymodel.set_bounds(eep=(1,1700), age=(6,11.))
    binarymodel.set_prior(distance=GaussianPrior(distance, 3*edist, 
                                                 bounds=(distance-6*edist, distance+6*edist)))
#     print(5.*mdist_diff, distance, 5.*pdist_diff)
    return binarymodel


def generate_params_for_interp(ticnum):
    
    ticrow = hq_joker_edr3_apogee_tess_df.loc[hq_joker_edr3_apogee_tess_df['ID'] == ticnum]
    
    tmass_errs = get_2mass_mag_uncertainties(ticrow)
    if tmass_errs.shape[0] > 0:
        assert tmass_errs.source_id.squeeze() == ticrow['source_id'].squeeze()
        eh,ej,ek = tmass_errs.eh.squeeze(), tmass_errs.ej.squeeze(), tmass_errs.ek.squeeze()
    if tmass_errs.shape[0] == 0:
        eh,ej,ek = 0.02,0.02,0.02


    g = ticrow.phot_g_mean_mag.squeeze()
    bp = ticrow.phot_bp_mean_mag.squeeze()
    rp = ticrow.phot_rp_mean_mag.squeeze()
    eg = (-2.5*np.log10(ticrow.phot_g_mean_flux.squeeze()-ticrow.phot_g_mean_flux_error.squeeze()) 
          + 2.5*np.log10(ticrow.phot_g_mean_flux.squeeze()+ticrow.phot_g_mean_flux_error.squeeze())) / 2.
    ebp = (-2.5*np.log10(ticrow.phot_bp_mean_flux.squeeze()-ticrow.phot_bp_mean_flux_error.squeeze()) 
          + 2.5*np.log10(ticrow.phot_bp_mean_flux.squeeze()+ticrow.phot_bp_mean_flux_error.squeeze())) / 2.
    erp = (-2.5*np.log10(ticrow.phot_rp_mean_flux.squeeze()-ticrow.phot_rp_mean_flux_error.squeeze()) 
          + 2.5*np.log10(ticrow.phot_rp_mean_flux.squeeze()+ticrow.phot_rp_mean_flux_error.squeeze())) / 2.
#     print(eg, ebp, erp)

    params = {
        'Teff':(ticrow.TEFF.squeeze(),ticrow.TEFF_ERR.squeeze()),
        'logg':(ticrow.LOGG.squeeze(),ticrow.LOGG_ERR.squeeze()),
        'feh':(ticrow.FE_H.squeeze(), ticrow.FE_H_ERR.squeeze()),
        'H':(ticrow.H.squeeze(), eh),
        'J':(ticrow.J.squeeze(), ej),
        'K':(ticrow.K.squeeze(), ek),
        'BP':(bp, ebp),
        'RP':(rp, erp),
        'G':(g, eg),
        # 'parallax':(ticrow.parallax.squeeze(), ticrow.parallax_error.squeeze())
    }
    
    return params


def get_best_age_eep_mass_bounds(TICNUM):

    interp_params = generate_params_for_interp(TICNUM)
    from isochrones.mist import MISTEvolutionTrackGrid
    track_grid = MISTEvolutionTrackGrid()
    eep_range = np.arange(1,1700,0.25)
    from tqdm import tqdm
    min_logg = interp_params['logg'][0] - 0.5
    max_logg = interp_params['logg'][0] + 0.5
    min_teff = interp_params['Teff'][0] - 500
    max_teff = interp_params['Teff'][0] + 500
    # print(tic_params['feh'])

    valid_ages, valid_eeps, valid_mass = [], [], []
    for masses in tqdm(np.linspace(0.1,50,1000), position=0, leave='None'):
        for eep in eep_range:
            interp_vals = track_grid.interp([interp_params['feh'][0], masses, eep], ['age','Teff','logg'])[0]
            age, teff, logg = interp_vals[0], interp_vals[1], interp_vals[2]
            if np.isfinite([age, teff, logg]).sum() == 3:
                bound1 = teff > min_teff
                bound2 = teff < max_teff
                bound3 = logg > min_logg
                bound4 = logg < max_logg
                if bound1 & bound2 & bound3 & bound4:
                    valid_ages.append(age)
                    valid_eeps.append(eep)
                    valid_mass.append(masses)

    return (valid_ages, valid_eeps, valid_mass)



def main(index=0,
        n_live_points=100,
        test_only=0):


    tic_systems_of_interest = [
    28159019,   # 0
    99254945,   # 1
    126232983,  # 2
    164458426,  # 3
    164527723,  # 4
    165453878,  # 5
    169820068,  # 6
    258108067,  # 7
    271548206,  # 8
    272074664,  # 9
    20215452,   # 10
    144441148,  # 11
    365204192   # 12
    ]
    ticsystem = tic_systems_of_interest[index]

    binmod = initialize_multinest_binary_model(ticsystem, TestOnly=test_only)

    # return

    # from mpi4py import MPI 
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # print(f"The rank on this machine is {rank}")

    valid_ages,valid_eeps,valid_mass = get_best_age_eep_mass_bounds(ticsystem)
    binmod.set_bounds(eep=(min(valid_eeps), max(valid_eeps)),
                  age=(min(valid_ages), max(valid_ages)),
                  mass=(max(0.1, min(valid_mass)), min(10, max(valid_mass)))
                 )
    print("Starting MultiNest")

    binmod.fit(n_live_points=n_live_points, overwrite=True)

    fig = corner(az.from_dict(
        binmod.derived_samples[['mass_0','mass_1','Mbol_0','Mbol_1','radius_0','radius_1']].to_dict('list')),
        smooth=1)
    fig.axes[0].set_title(f'TIC-{ticsystem}')
    plt.savefig(f"{DD}tic_{ticsystem}_multinest_corner.png",dpi=150,bbox_inches='tight')
    plt.close(fig)


    m0,m1,r0,r1,mbol0,mbol1 = binmod.derived_samples[['mass_0','mass_1','radius_0', 'radius_1','Mbol_0','Mbol_1']].values.T
    num, denom = np.argmin([np.median(m0), np.median(m1)]), np.argmax([np.median(m0), np.median(m1)])

    ms, mp = [m0,m1][num], [m0,m1][denom]

    rs, rp = [r0,r1][num], [r0,r1][denom]
    
    mbols,mbolp = [mbol0,mbol1][num], [mbol0,mbol1][denom]
    
    binmod.derived_samples['logMp'] = np.log(mp)
    binmod.derived_samples['logRp'] = np.log(rp)
    binmod.derived_samples['logk'] = np.log(rs / rp)
    binmod.derived_samples['logq'] = np.log(ms / mp)
    binmod.derived_samples['logs'] = np.log(mbols / mbolp)

    binmod.save_hdf(f"{DD}ceph/pymultinest_fits/tic_{ticsystem}_binary_model_obj.hdf",overwrite=True)
    
    fig = corner(az.from_dict(binmod.derived_samples[['logMp','logRp','logk','logq','logs']].to_dict('list')))
    fig.axes[0].set_title(ticsystem) 
    plt.savefig(f"{DD}tic_{ticsystem}_pymc3_priors_corner.png",dpi=150, bbox_inches='tight')
    plt.close(fig)


result = OptionParser()

result.add_option('--index', dest='index', default=0, type='int', 
                help='indice of tic system array (defaults to 0)')
result.add_option("--n_live_points", dest='n_live_points', default=100, type='int',
                help='number of live points to run multinest (default: 100)')
result.add_option("--test_only", dest="test_only", default=0, type='int',
                help='run test isochrones run with t added to output file')
if __name__ == "__main__":
    opt,arguments = result.parse_args()
    main(**opt.__dict__)



# for ticsystem in tic_systems_of_interest:
#     binmod = initialize_multinest_binary_model(ticsystem)

#     valid_ages,valid_eeps,valid_mass = get_best_age_eep_mass_bounds(ticsystem)
#     binmod.set_bounds(eep=(min(valid_eeps), max(valid_eeps)),
#                   age=(min(valid_ages), max(valid_ages)),
#                   mass=(max(0.1, min(valid_mass)), min(10, max(valid_mass)))
#                  )
#     print("Starting MultiNest")
#     binmod.fit(n_live_points=2000, overwrite=True)
#     fig = corner(az.from_dict(
#         binmod.derived_samples[['mass_0','mass_1','Mbol_0','Mbol_1','radius_0','radius_1']].to_dict('list')),
#         smooth=1)
#     fig.axes[0].set_title(f'TIC-{ticsystem}')
#     plt.savefig(f"/Users/kjaehnig/CCA_work/GAT/figs/tic_{ticsystem}_multinest_corner.png",dpi=150,bbox_inches='tight')
#     plt.close()
#     binmod.save_hdf(f"/Users/kjaehnig/CCA_work/GAT/pymultinest_fits/tic_{ticsystem}_binary_model_obj.hdf",overwrite=True)





