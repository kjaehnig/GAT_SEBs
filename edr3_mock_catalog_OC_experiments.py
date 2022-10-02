#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import math as ma
import pyvo as pvy
import astropy as ast
import matplotlib.pyplot as plt
from astroquery.utils.tap.core import TapPlus
from collections import Counter
from scipy.stats import epps_singleton_2samp as es2
import scipy
import sklearn as skl
from tqdm import tqdm
import pickle as pk
from optparse import OptionParser
from sklearn.preprocessing import RobustScaler
from xdgmm import XDGMM
import helper_functions as hf
import os
from astropy import units as u
# get_ipython().run_line_magic('pinfo', 'pvy.tablesearch')

def add_obs_err_to_mock(clstdat,plx_factor=1.3):
    clst = clstdat.copy()
    
    columns = [
        'ra',
        'dec',
        'parallax',
        'pmra',
        'pmdec',
    ]

    
    for col in columns:
        tru_col,tru_col_err = clst[col],clst[col+'_error']
        
        if col=='parallax':
            tru_col_err = 10**((np.log10(tru_col_err) + 1)*plx_factor)
            
        obs_col = np.random.normal(loc=tru_col,
                                   scale=tru_col_err,
                                   size=(1000,tru_col.shape[0])
                                  )
        clst[col+'_error'] = obs_col.std(axis=0)
        clst[col] = obs_col.mean(axis=0)
        
    return clst

columns = [
    'ra',
    'dec',
    'parallax',
    'pmra',
    'pmdec',
]

error_columns = [
    'ra_error',
    'dec_error',
    'parallax_error',
    'pmra_error',
    'pmdec_error',
]

corr_map = {
    'ra_dec_corr': [0, 1],
    'ra_parallax_corr': [0, 2],
    'ra_pmra_corr': [0, 3],
    'ra_pmdec_corr': [0, 4],
    'dec_parallax_corr': [1, 2],
    'dec_pmra_corr': [1, 3],
    'dec_pmdec_corr': [1, 4],
    'parallax_pmra_corr': [2, 3],
    'parallax_pmdec_corr': [2, 4],
    'pmra_pmdec_corr': [3, 4]
}


def assemble_gaia_covariance_matrix(df, scalings=None):
    X = df[columns].fillna(0.0).values
    C = np.zeros((len(df), 5, 5))
    diag = np.arange(5)
    if scalings is None: scalings = np.ones_like(df[error_columns].values)
    C[:, diag, diag] = df[error_columns].fillna(1e6).values/scalings

    for column, (i, j) in corr_map.items():
        C[:, i, j] = df[column].fillna(0).values
        C[:, i, j] *= (C[:, i, i] * C[:, j, j])
        C[:, j, i] = C[:, i, j]

    C[:, diag, diag] = C[:, diag, diag]**2

    return X, C


def scale_covariance_matrices(cov_arr, scalings):
    diag = np.arange(5)
    C = cov_arr.copy()

    for col, (i,j) in corr_map.items():
        C[:, i, j] /= abs((scalings[i]*scalings[j]))

    C[:, diag, diag] /= scalings**2.

    return C


def assemble_scaling_matrix(scalings):
    diag = np.arange(5)
    C = np.zeros((len(scalings),len(scalings)))
    C[diag,diag] = scalings
    for column, (i, j) in corr_map.items():
        C[i, j] = (C[i, i] * C[j, j])
        C[j, i] = C[i, j]

    C[diag,diag] = C[diag,diag]**2.

    return C
# In[16]:

from tqdm import tqdm
def bootstrap_synthetic_covariance_matrix(X,C,N):
    
    X_cp = np.zeros_like(X)
    C_cp = np.zeros_like(C)
    for i_s in tqdm(range(X.shape[0])):
        synth_draws = np.random.multivariate_normal(mean=X[i_s], cov=C[i_s], size=N)
        X_cp[i_s] = synth_draws.mean(axis=0)
        C_cp[i_s,:,:] = np.cov(synth_draws.T) 
    return (X_cp, C_cp)

# In[5]:

def resample_from_onsky_pts(random_pts=None,number=10):
    random_indices = np.random.choice(len(random_pts[0]),size=number,replace=False)
    return (random_pts[0][random_indices],random_pts[1][random_indices])


# print(CG2020clsts.sort_values('N',ascending=False).head())
def weighted_MAD(x,w):
    """ Takes a variable and its weights to calculate the weighted absolute
    median deviation as a replacement for the standard deviation.
    Parameters
    ----------
        x : array
            Variable of interest
        w : array
            Weights associated with variable of interest.
    Returns
    -------
        weightedMAD : float
            The weighted median absolute deviation of the variable of
            interest.
    """
    import weighted
    center_median = weighted.median(x,w)
    rescaled_x = abs(x-center_median)
    weightedMAD = weighted.median(rescaled_x, w)
    return weightedMAD


from astropy.coordinates import SkyCoord
def get_random_onsky_pts(FOV_DAT=None,size=100):
    allpos = SkyCoord(ra=FOV_DAT.ra, 
                    dec=FOV_DAT.dec,
                    unit=u.deg,
                    frame='icrs')

    clstcntr = SkyCoord(ra=FOV_DAT.ra.median(),
                        dec=FOV_DAT.dec.median(),
                        unit=u.deg,
                        frame='icrs')

    radLOS = clstcntr.separation(allpos).to(u.deg).value.max()

    #random_pts = sample_uniform_sphere_centered(clstpos, 
    #                                        RADIUS=radLOS,
    #                                        encompass_circle=True,
    #                                       size=size)
    uni_ra = np.random.uniform(low=FOV_DAT.ra.min(),
                                high=FOV_DAT.ra.max(),
                                size=size*2)
    uni_dec = np.random.uniform(low=FOV_DAT.dec.min(),
                                high=FOV_DAT.dec.max(),
                                size=size*2)

    rndmpos = SkyCoord(ra=uni_ra,
                        dec=uni_dec,
                        unit=u.deg,
                        frame='icrs')
    rndmFOVpos = clstcntr.separation(rndmpos).to(u.deg).value
    return (uni_ra[rndmFOVpos <= radLOS],
                uni_dec[rndmFOVpos <= radLOS])


def get_mad_sigmaclip_params(cdata):
    """ Takes a cluster member stars dataframe and calculates the weighted 
    median absolute deviation and the associated weighted uncertainty for
    [ra, dec, parallax, pmra, pmdec] using the module 'weighted'.
    
    Parameters
    ---------- 
        cdata : pandas dataframe
            Star cluster member stars individual measurements
    Returns
    -------
    """
    import weighted

    params = ['ra','dec','parallax','pmra','pmdec']
    initial_samp = cdata#.loc[cdata.xdg_proba>=0.85]

    weighted_median_res = []
    weighted_MAD_res = []

    for cols in params:
        cdata_col = initial_samp[cols]
        cdata_ecol = initial_samp[cols+'_error']
        initial_median = weighted.median(cdata_col,1./cdata_ecol)
        initial_mad = weighted_MAD(cdata_col,1./cdata_ecol)
        #print(initial_median, initial_mad)
        MAD3 = 1.4826 * initial_mad * 3.

        truncated_samp = initial_samp.loc[
                            cdata_col <= initial_median +  MAD3].loc[
                            cdata_col >= initial_median - MAD3]

        final_median = weighted.median(truncated_samp[cols],
                                    1./truncated_samp[cols+'_error'])
        final_MAD = weighted_MAD(truncated_samp[cols],
                            1./truncated_samp[cols+'_error'])

        weighted_median_res.append(final_median)
        weighted_MAD_res.append(final_MAD * 1.4826)
    return weighted_median_res, weighted_MAD_res


def compute_proper_motion_dispersion_check(cls_dat):

    params_mean,params_sig = get_mad_sigmaclip_params(cls_dat)
    med_plx = params_mean[2]    

    cand_spmpm = np.sqrt(params_sig[3]**2 + 
                        params_sig[4]**2)

    if med_plx <= 1.0:
        spmpm_lim = 1.0
    if med_plx > 1.0:
        spmpm_lim = 5.*np.sqrt(2)*(med_plx/4.74) # emperical limit from CG asterism paper

    if cand_spmpm < spmpm_lim:
        return  1

    if cand_spmpm > spmpm_lim:
        return 0


def get_mst_branch_sums(pts):
    import sklearn
    alpha = pts[0]
    delta = pts[1]
    coords = np.array([delta,alpha]).T
    assert coords.shape[0] == len(alpha)
    distmatr = sklearn.metrics.pairwise.haversine_distances(
                                        np.deg2rad(coords))
    csgraph = scipy.sparse.csc_matrix(distmatr)
    mintree = scipy.sparse.csgraph.minimum_spanning_tree(csgraph)
    sum_branch_lengths = np.sum(mintree.todense().flatten())
    return sum_branch_lengths


def check_cluster_spatial_proper_motion_spread(cls_dat,fov_dat):

    spmpm_check = compute_proper_motion_dispersion_check(cls_dat)

    asmr_check = run_mst_asmr_check(cls_dat, fov_dat)

    if asmr_check & spmpm_check:
        return 1
    else:
        return 0


def run_mst_asmr_check(cls_dat, fov_dat):
    rndm_pts = get_random_onsky_pts(fov_dat, size=10000)

    cand_ra,cand_de = cls_dat[['ra','dec']].values.T

    cand_lobs = get_mst_branch_sums((cand_ra,
                                    cand_de))

    l,sigl = get_avg_mst_branch_lengths(
                    rndm_pts,
                    size=cand_ra.shape[0],
                    niter=500)
    
    asmr = (l-cand_lobs)/sigl

    if asmr > 1:
        return 1
    if asmr < 1:
        return 0


def get_avg_mst_branch_lengths(pts=None,size=10,niter=10):
    #sampled_pts = resample_from_onsky_pts(pts,number=size)
    
    mst_res = np.zeros(niter)
    for ii in tqdm(range(niter),position=3,leave=None):
        sampled_pts = resample_from_onsky_pts(pts,number=size)
        sum_branches = get_mst_branch_sums(sampled_pts)
        mst_res[ii] = sum_branches   
    #print('\033[1A')    
    return np.mean(mst_res), np.std(mst_res)


def get_stars_in_fov(maxN, RA, DEC, PMRA, PMDE, R50, Plx, return_dat=False):
    """ 
    I would inner join to gaiadr3.astrophysical_parameters
    left outer join to gaiadr3.nss_two_body_orbit
    left outer join to gaiadr3.binary_masses
    Parameters
    ----------
    Returns
    -------
    """
    qry_cols=['source_id','ra', 'ra_error',
        'dec', 'dec_error', 'parallax', 'parallax_error',
        'parallax_over_error', 'pmra', 'pmra_error',
        'pmdec', 'pmdec_error','ra_dec_corr',
        'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
        'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
        'parallax_pmra_corr', 'parallax_pmdec_corr',
         'pmra_pmdec_corr','phot_g_mean_mag','bp_rp']

    # columns = ['gdr3.'+ii+', ' for ii in columns]
    qry_col_str = ['gdr3.'+ii+', ' for ii in qry_cols]
    qry_col_str = ''.join(qry_col_str)

    from astroquery.gaia import Gaia 
    from astropy.table import Table

    Gaia.login(user='kjaehnig',
                password='Legacyofash117!', verbose=True)


    res = Gaia.launch_job_async(query=f"select TOP {maxN} gdr3.source_id\
                FROM gaiadr3.gaia_source_lite as gdr3 \
                WHERE gdr3.parallax_over_error > 5 \
                AND gdr3.parallax < {Plx + 1} \
                AND gdr3.parallax > {Plx - 1} \
                AND DISTANCE(gdr3.ra, gdr3.dec, {RA},{DEC}) < {R50} \
                "
            )
                # SQRT(POWER(gdr3.pmra - {PMRA},2) + POWER(gdr3.pmdec - {PMDE},2)) as pmpmdist \
    this_jobs_id = res.jobid
    dat = res.get_results().to_pandas() 
    Gaia.remove_jobs(this_jobs_id)
    # dat['Cluster'] = dat.Cluster.str.decode("UTF-8")
    # print(dat.dropna().shape)
    if return_dat:
        return dat
    else:
        return dat.shape[0]


def main(index, 
        part2=0,
        testrun=1):

    if testrun:
        print("Running in test mode.")
        tol_val, regularization_val = 1e-6, 1e-8
    else:
        print("Running in production mode.")
        tol_val, regularization_val = 1e-8, 1e-10

    print(skl.__version__)
    if part2:
        index += 1000
    DD = hf.load_system_specific_directory()
    if not os.path.exists(DD+"failed_xdgmm_mocks"):
        os.mkdir(DD+"failed_xdgmm_mocks/")

    max_rec = 20000
    # clst_name = 'NGC_7789'
    CG2020clsts = pd.read_csv("cantat_gaudin_2020_cluster_catalog.csv")
    # CG2020membs = pd.read_csv("cantat_gaudin_2020_member_catalog.csv")

    clst_name = CG2020clsts.iloc[index]['Cluster']

    clstqry = CG2020clsts.loc[CG2020clsts.Cluster == clst_name]
    # membqry = CG2020membs.loc[CG2020membs.Cluster == clst_name]
    # Nmembs = membqry.loc[membqry.Proba > 0.5].shape[0]
    # print('CG2020 cluster membership has: ', Nmembs)
    print(clstqry[['Cluster','Rgc','N','Plx','r50']])
    # print(clstqry.T)

    RA,DEC,PMRA,PMDE,R50,Plx = (
        clstqry.RA_ICRS.squeeze(),
        clstqry.DE_ICRS.squeeze(),
        clstqry.pmRA.squeeze(),
        clstqry.pmDE.squeeze(),
        5*clstqry.r50.squeeze(),
        clstqry.Plx.squeeze()
    )

    qry_cols=['source_id','ra', 'ra_error',
        'dec', 'dec_error', 'parallax', 'parallax_error',
        'pmra', 'pmra_error',
        'pmdec', 'pmdec_error','ra_dec_corr',
        'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
        'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
        'parallax_pmra_corr', 'parallax_pmdec_corr',
         'pmra_pmdec_corr','phot_g_mean_mag','bp_rp']

    qry_cols = [ii+', ' for ii in qry_cols]

    qry_col_str = ''.join(qry_cols)

    tap_url = "http://dc.zah.uni-heidelberg.de/tap"
    tap_oc_query = f"SELECT *  \
                    FROM gedr3mock.main \
                    WHERE popid = 11  \
                    AND ABS(parallax - {Plx}) < 2 \
                    AND distance({RA}, {DEC}, ra, dec) < {R50}"

    tap_fs_query = f"select * \
                     FROM gedr3mock.main WHERE popid != 11  \
                     AND d11y > 99 \
                     AND 1 = CONTAINS(POINT({RA},{DEC})\
                         ,CIRCLE(gedr3mock.main.ra, gedr3mock.main.dec,{R50}))"

    # print(tap_oc_query)
    
    failed_tries = 0
    query_attempts = 10
    Nfov_dr3 = None
    try:
        print("querying DR3 for Nstars in FOV")
        Nfov_dr3 = get_stars_in_fov(int(max_rec/2), RA, DEC, PMRA, PMDE, R50, Plx, return_dat=False)
    except:
        print("dr3 query failed or timed out")
        Nfov_dr3 = None
    for ntry in range(query_attempts):
        try: 
            pvy_cs = pvy.tablesearch(url=tap_url, query=tap_oc_query, maxrec=max_rec)
            pvy_fs = pvy.tablesearch(url=tap_url, query=tap_fs_query, maxrec=max_rec)
            clsts = pvy_cs.to_table().to_pandas()
            clsts['cluster_flag'] = np.ones(clsts.shape[0])
            flds = pvy_fs.to_table().to_pandas()
            flds['cluster_flag'] = np.zeros(flds.shape[0])
            print(f"Successful mock-catalog query after {failed_tries} failures")
            break
        except:
            failed_tries += 1

    if failed_tries == query_attempts:
        file = open(DD+f"failed_xdgmm_mocks/{clst_name}_FAILED_GAVO_QUERY",'wb')
        file.close()
        return

    if clsts.shape[0] < 12:
        file = open(DD+f"failed_xdgmm_mocks/{clst_name}_Nclst_LT12",'wb')
        file.close()
        return


    # clsts = clsts[clsts['parallax']/clsts['parallax_error']] > 10
    # clsts = flds[flds['parallax']/flds['parallax_error']] > 10



    # usXdr3, usCdr3 = assemble_gaia_covariance_matrix(dr3_dat)
    # dr3scaler = RobustScaler().fit(usXdr3)
    # dr3scalings = dr3scaler.scale_

    # Xdr3 = dr3scaler.transform(usXdr3)
    # Cdr3 = scale_covariance_matrices(usCdr3, dr3scalings)
    # cov_scaler = assemble_scaling_matrix(dr3scalings)
    
    # Cdr3 = usCdr3 / cov_scaler

    # pos_def = []
    # for cov in Cdr3:
    #     try:
    #         np.linalg.cholesky(cov)
    #         pos_def.append(1)
    #     except:
    #         pos_def.append(0)

    # print(f"there are {sum(pos_def)}/{len(pos_def)} positive-definite covariance matrices")

    # dr3mod = XDGMM(tol=1e-8,
    #             method='Bovy',
    #             n_iter=10**9,
    #             n_components=2,
    #             random_state=999,
    #             w = np.min(usCdr3/cov_scaler)**2.)


    # dr3mod.fit(Xdr3, Cdr3)
    # try:
    #     np.linalg.cholesky(dr3mod.V[0])
    # except:
    #     print(f"component 0 is NOT PosSemiDef")
    # try:
    #     np.linalg.cholesky(dr3mod.V[1])
    # except:
    #     print(f"component 1 is NOT PosSemiDef")
    #     return
    # compV = dr3mod.V
    # compDE = (5./2) + (5./2.)*np.log(2.*np.pi) + .5*np.log(np.linalg.det(compV))

    # cluster_lbl, field_lbl = np.argmin(compDE), np.argmax(compDE)

    # dr3proba = dr3mod.predict_proba(Xdr3, Cdr3)
    # joint_proba = dr3proba[:,cluster_lbl]# * (1 - dr3proba[:,field_lbl])

    # dr3labels = np.zeros_like(joint_proba)
    # dr3labels[joint_proba > 0.5] = 1.0


    # Ncluster = int(np.sum(dr3labels))
    # Nfield = int(len(dr3labels) - Ncluster)


    # print(f"finished running XDGMM on fov for {clst_name}.")
    # print(f"XDGMM classified {Ncluster} OC stars and {Nfield} field stars")
    # print(F"starting XDGMM fit for mock catalog FOV.")
    ### -----------------------------------------------------------------------

    # print("Inflating plx_error to better approximate Gaia DR3")
    # clsts = add_obs_err_to_mock(clsts)
    if clsts.shape[0] > clstqry.N.squeeze():
        print("Down-sampling mock cluster")
        clsts = clsts.sample(clstqry.N.squeeze())
    if (Nfov_dr3 is not None):
        if (flds.shape[0] > Nfov_dr3) & (Nfov_dr3 < int(max_rec/2.)):
            print("Down-sampling mock field FOV using DR3 FOV")
            flds = flds.sample(Nfov_dr3)
    if (Nfov_dr3 is None):
        if (flds.shape[0] > 2500) & (clstqry.N.squeeze() < 1000):
            print("Down-sampling mock field FOV to 2500 stars")
            flds = flds.sample(2500)


    Nflds_still_too_large = flds.shape[0] > 2500
    if Nflds_still_too_large:
        flds = flds.sample(2500)

    print("N mock cluster stars:  ",clsts.shape[0])
    print("N mock field stars:    ",flds.shape[0])

    fov_ = pd.concat([clsts, flds], ignore_index=True)
    fov_ = fov_.sort_values('logg')

    fov_obs = fov_.copy()

    X,C = assemble_gaia_covariance_matrix(fov_)
    usXcp,usCcp = bootstrap_synthetic_covariance_matrix(X,C,10000)

    obs_cols = ['ra','dec','parallax','pmra','pmdec']
    for ii,col in enumerate(obs_cols):
        fov_obs[col] = usXcp[:,ii]        ##### USE THIS FOV FILE TO COMPARE OBS TO MODEL

    scaler = RobustScaler().fit(usXcp)
    scalings_ = scaler.scale_

    Xcp =  scaler.transform(usXcp)
    # Ccp =  scale_covariance_matrices(usCcp, scalings_)
    cov_scaler = assemble_scaling_matrix(scalings_)
    
    Ccp = usCcp / cov_scaler
    
    print("Starting XDGMM run.")

    xdmod = XDGMM(tol=tol_val, 
                method='Bovy', 
                n_iter=10**9, 
                n_components=2, 
                random_state=666,
                w=regularization_val)

    bic_test_failure_flag = 1
    opt_Nc = 4
    try:
        bics, opt_Nc, min_bic = xdmod.bic_test(
                                        Xcp, Ccp,
                                        param_range=np.arange(0,9)+2,
                                        no_err = False
                                    )
        print(f"best model (acc. to BIC) is: {opt_Nc}")
        bic_test_failure_flag = 0
    except:
        print("Model failed during XDGMM BIC test")
        print(f"Setting Model N_c to default {opt_Nc}")
    try:
        xdmod = XDGMM(tol=tol_val, 
                method='Bovy', 
                n_iter=10**9, 
                n_components=opt_Nc, 
                random_state=666,
                w=regularization_val)

        xdmod.fit(Xcp, Ccp)
        
        proba = xdmod.predict_proba(Xcp, Ccp)
        
        per_component_labels = proba.argmax(axis=1)

        def compute_diff_entp(V):
            DE = (V.shape[0]/2.) \
                  * (1. + np.log(2. * np.pi)) \
                  + .5 * np.log(np.linalg.det(V))
            return DE

        N_per_comp = np.array([sum(proba[:,ii] > 0.5) for ii in range(opt_Nc)])
        sorted_by_Ncomp = np.argsort(N_per_comp)[::-1]

        DEs = np.array([compute_diff_entp(Vi) for Vi in xdmod.V])

        best_comp = np.inf
        min_DE = np.inf
        min_delpm2 = np.inf
        for i_c in sorted_by_Ncomp:
            cand_probs = proba[:,i_c]

            memb_mask = cand_probs > 0.5
            cand_dat = fov_obs[memb_mask]
            act_dat = fov_obs[fov_obs['popid'] == 11]
            # print(sum(memb_mask))
            isnt_asterism = check_cluster_spatial_proper_motion_spread(cand_dat, fov_obs)
            # lit_param = np.array([RA,DEC,Plx,PMRA,PMDE]).reshape(1,-1)
            # lra,ldec,lplx,lpmra,lpmde = scaler.transform(lit_param).squeeze()

            # cand_param, _ = get_mad_sigmaclip_params(cand_dat)
            # act_param, _ = get_mad_sigmaclip_params(act_dat)
            # cand_delpm2 = np.sqrt( (act_param[3] - cand_param[3])**2 + 
            #                     (act_param[4] - cand_param[4])**2 )

            lesser_de = DEs[i_c] < min_DE
            suff_memb = sum(memb_mask) > 12
            # closer_cntr = cand_delpm2 < min_delpm2
            
            print(i_c, sum(memb_mask),lesser_de, suff_memb, isnt_asterism)

            if lesser_de & suff_memb & isnt_asterism:
                print(f"comp-{i_c} might be a cluster")
                min_DE = DEs[i_c]
                best_comp = i_c
                # min_delpm2 = cand_delpm2
        print(f"Component most likely to be cluster is comp-{best_comp}")
        cluster_lbl = best_comp

        if cluster_lbl == np.inf:
            file = open(f"failed_xdgmm_mocks/{clst_name}_XDGMM_FOUND_NO_OCs",'wb')
            file.close()
            return
        #cluster_lbl, field_lbl = np.argmin(compDE), np.argmax(compDE)

    except:
        file = open(f"failed_xdgmm_mocks/{clst_name}_XDGMM_FAILED",'wb')
        file.close()
        return

    joint_proba = proba[:,cluster_lbl]# * (1 - proba[:,field_lbl])
        # file = open(f"{clst_name}_xdgmm_failed",'wb')
        # file.close()

    # clst_mask = proba[:,1] > 0.75
    # labels = np.argmax(proba,axis=1)


    mocklabels = np.zeros_like(joint_proba)
    mocklabels[joint_proba > 0.5] = 1.0
    # lbl_act[labels==clst] = 1
    # lbl_act[labels!=clst] = 0
    print(Counter(mocklabels))
    BAS = skl.metrics.balanced_accuracy_score(fov_['cluster_flag'], mocklabels)

    APS = skl.metrics.average_precision_score(fov_['cluster_flag'], mocklabels,
            average='weighted')
    
    CM = skl.metrics.confusion_matrix(fov_['cluster_flag'],mocklabels)

    CR = skl.metrics.classification_report(fov_['cluster_flag'],mocklabels,
            output_dict=True,target_names=['field','cluster'])

    ROC_AUC = skl.metrics.roc_auc_score(fov_['cluster_flag'],mocklabels,
                average='weighted')

    CR['Cluster'] = clst_name
    CR['mock_fov'] = fov_
    CR['scaler'] = {'type':'Robust',
                    'scale_':scalings_,
                    'center_':scaler.center_}
    
    CR['best_modelcomp'] = str(opt_Nc).zfill(2)+str(cluster_lbl).zfill(2)
    CR['bic_test_failure_flag'] = bic_test_failure_flag

    CR['labels'] = mocklabels
    CR['confusion_matrix'] = CM 
    CR['TnFpFnTp'] = CM.ravel()
    CR['balanced_accuracy_score'] = BAS
    CR['average_precision_score'] = APS
    CR['roc_auc_score'] = ROC_AUC
    # return precision_recall_fscore_support
    if not os.path.exists(DD+"xdgmm_performance_dicts"):
        os.mkdir(DD+"xdgmm_performance_dicts")
    file = open(DD+f"xdgmm_performance_dicts/{clst_name}_xdgmm_performance_dict.pk",'wb')
    pk.dump(CR, file)
    file.close()
# cluster_xdgmm_performance = {'clsts':[],
#                                 'precision_recall_fscore_support':[]
#                                 }

# for clsts in tqdm(CG2020clsts.Cluster):

#     cluster_xdgmm_performance['clsts'].append(clsts)
#     try:
#         prfs = main(clsts)

#         cluster_xdgmm_performance['precision_recall_fscore_support'].append(prfs)
#     except:
#         print('model failed')

# file = open("xdgmm_performance.pk",'wb')
# pk.dump(cluster_xdgmm_performance, file)
# file.close()



result = OptionParser()
result.add_option('-i', dest='index', default=0, type='int', 
                help='indice of cluster to fit from CG2020')
result.add_option('--p2',dest='part2',default=0, type='int',
                help='arg to increased indices wo angering slurm')
result.add_option('--tr',dest='testrun', default=1, type='int',
                help='if 1, then lower tol for faster testing')
if __name__ == "__main__":
    opt,arguments = result.parse_args()
    main(**opt.__dict__)
# In[443]:


# import astroML


# In[ ]:







