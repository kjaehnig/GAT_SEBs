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
import sklearn as skl
from tqdm import tqdm
import pickle as pk
from optparse import OptionParser
from sklearn.preprocessing import RobustScaler
from xdgmm import XDGMM
import helper_functions as hf
import os
# get_ipython().run_line_magic('pinfo', 'pvy.tablesearch')


# In[5]:


# print(CG2020clsts.sort_values('N',ascending=False).head())




def get_stars_in_fov(maxN, RA, DEC, PMRA, PMDE, R50, Plx):
    """ 
    I would inner join to gaiadr3.astrophysical_parameters
    left outer join to gaiadr3.nss_two_body_orbit
    left outer join to gaiadr3.binary_masses
    Parameters
    ----------
    Returns
    -------
    """


    from astroquery.gaia import Gaia 
    from astropy.table import Table

    Gaia.login(user='kjaehnig',
                password='Legacyofash117!', verbose=True)


    res = Gaia.launch_job_async(query=f"select TOP {maxN} {col_str} \
                SQRT(POWER(gdr3.pmra - {PMRA},2) + POWER(gdr3.pmdec - {PMDE},2)) as pmpmdist \
                FROM gaiadr3.gaia_source as gdr3 \
                WHERE gdr3.parallax_over_error > 5 \
                AND gdr3.parallax < {Plx + 0.75} \
                AND gdr3.parallax > {Plx - 0.75} \
                AND DISTANCE(gdr3.ra, gdr3.dec, {RA},{DEC}) < {R50} \
                "
            )

    # dat = res.get_results().to_pandas() 
    # dat['Cluster'] = dat.Cluster.str.decode("UTF-8")
    # print(dat.dropna().shape)

    return res

def main(index):
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
        2*clstqry.r50.squeeze(),
        clstqry.Plx.squeeze()
    )
    columns=['source_id','ra', 'ra_error',
        'dec', 'dec_error', 'parallax', 'parallax_error',
        'parallax_over_error', 'pmra', 'pmra_error',
        'pmdec', 'pmdec_error','ra_dec_corr',
        'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
        'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
        'parallax_pmra_corr', 'parallax_pmdec_corr',
         'pmra_pmdec_corr','phot_g_mean_mag','bp_rp','popid']

    # columns = ['gdr3.'+ii+', ' for ii in columns]
    columns = [ii+', ' for ii in columns[:-2]]
    col_str = ''.join(columns)

    tap_url = "http://dc.zah.uni-heidelberg.de/tap"
    tap_oc_query = f"select * \
                     FROM gedr3mock.main WHERE popid = 11  \
                     AND parallax/parallax_error > 10 \
                     AND ABS(parallax - {Plx}) < 2 \
                     AND 1 = CONTAINS(POINT({RA}, {DEC})\
                         ,CIRCLE(gedr3mock.main.ra, gedr3mock.main.dec,{R50}))"

    tap_fs_query = f"select * \
                     FROM gedr3mock.main WHERE popid != 11  \
                     AND parallax/parallax_error > 10 \
                     AND ABS(parallax - {Plx}) <  2 \
                     AND 1 = CONTAINS(POINT({RA},{DEC})\
                         ,CIRCLE(gedr3mock.main.ra, gedr3mock.main.dec,{R50}))"

    print(tap_oc_query)
    
    try: 
        # dr3_dat = get_stars_in_fov(int(max_rec/2.), RA, DEC, PMRA, PMDE, R50, Plx)
        # print('stars in dr3 FOV: ',dr3_dat.shape[0])
        pvy_cs = pvy.tablesearch(url=tap_url, query=tap_oc_query,maxrec=max_rec)
        pvy_fs = pvy.tablesearch(url=tap_url,query = tap_fs_query, maxrec=max_rec)
        clsts = pvy_cs.to_table().to_pandas()
        clsts['cluster_flag'] = np.ones(clsts.shape[0])
        flds = pvy_fs.to_table().to_pandas()
        flds['cluster_flag'] = np.zeros(flds.shape[0])
    except:
        file = open(DD+f"failed_xdgmm_mocks/{clst_name}_FAILED_GAVO_QUERY",'wb')
        file.close()
        return

    if clsts.shape[0] < 12:
        file = open(DD+f"failed_xdgmm_mocks/{clst_name}_Nclst_LT12",'wb')
        file.close()
        return


    # clsts = clsts[clsts['parallax']/clsts['parallax_error']] > 10
    # clsts = flds[flds['parallax']/flds['parallax_error']] > 10
    def add_obs_err_to_mock(clstdat,plx_factor=1.1):
        clst = clstdat.copy()
        
        columns = [
         'ra','dec',
            'pmra','pmdec',
            'parallax'
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


    def assemble_gaia_covariance_matrix(df):
        X = df[columns].fillna(0.0).values
        C = np.zeros((len(df), 5, 5))
        diag = np.arange(5)
        C[:, diag, diag] = df[error_columns].fillna(1e6).values

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
    def bootstrap_synthetic_covariance_matrix(X,C):
        
        X_cp = np.zeros_like(X)
        C_cp = np.zeros_like(C)
        for i_s in tqdm(range(X.shape[0])):
            synth_draws = np.random.multivariate_normal(mean=X[i_s], cov=C[i_s], size=1000)
            X_cp[i_s] = synth_draws.mean(axis=0)
            C_cp[i_s,:,:] = np.cov(synth_draws.T) 
        return (X_cp, C_cp)


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



    fov_ = pd.concat([clsts, flds], ignore_index=True)

    X,C = assemble_gaia_covariance_matrix(fov_)
    usXcp,usCcp = bootstrap_synthetic_covariance_matrix(X,C)

    scaler = RobustScaler().fit(usXcp)
    scalings_ = scaler.scale_

    Xcp =  scaler.transform(usXcp)
    # Ccp =  scale_covariance_matrices(usCcp, scalings_)
    cov_scaler = assemble_scaling_matrix(scalings_)
    
    Ccp = usCcp / cov_scaler
    
    xdmod = XDGMM(tol=1e-8, 
                method='Bovy', 
                n_iter=10**9, 
                n_components=2, 
                random_state=666,
                w=np.min(Ccp)**2.)
    try:
        xdmod.fit(Xcp, Ccp)

        compV = xdmod.V
        compDE = (5./2) + (5./2.)*np.log(2.*np.pi) + .5*np.log(np.linalg.det(compV))

        cluster_lbl, field_lbl = np.argmin(compDE), np.argmax(compDE)

        proba = xdmod.predict_proba(Xcp, Ccp)
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
    CR['labels'] = mocklabels
    CR['confusion_matrix'] = CM 
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

if __name__ == "__main__":
    opt,arguments = result.parse_args()
    main(**opt.__dict__)
# In[443]:


# import astroML


# In[ ]:




