import scipy
import numpy as np, random
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import time
import pylab as pl
from IPython import display
from collections import defaultdict
import pandas as pd, seaborn as sns
import os
from scipy.stats import beta
import datetime
import json
from tqdm import tqdm


def get_served_ad_list(temp):
    ad_betas = list(zip(temp.clicks, temp.non_clicks))
    num_impr = temp['time_imps'].sum()
    print(num_impr)
    #num_impr = 10000
    k = len(ad_betas)
    A = np.zeros((num_impr, k), dtype='float32')
    #print_every = 1000
    for idx, (a, b) in enumerate(ad_betas):
        sampled_ctr = beta.rvs(a, b, size=num_impr)
        A[:, idx] = sampled_ctr
    #     if idx%print_every==0:
    #         print(f"{idx+1} of {k}.")
    served = np.argmax(A, axis=1)
    return served


def get_ad_fq(served_ad_list):
    fq_dict = {}
    for idx in served_ad_list:
        fq_dict.get(idx,0)
        fq_dict[idx] = fq_dict.get(idx,0)+1
    return fq_dict


def mab_imps_simulator_v1(mds_agg_data_path):
    mds_id = pd.read_csv(mds_agg_data_path)
    bubly_ads = [6376,6377, 6378]
    t = mds_id[(mds_id['date'] == '2021-06-16') & (mds_id['ad_id'].isin(bubly_ads))]
    t = t.groupby(['ad_id', 'metadata_dsrow_id'])['imp', 'clicks'].agg(sum).reset_index()
    t['non_clicks'] = t['imp'] - t['clicks']
    t['ctr'] = (t['clicks']/t['imp'])*100
    timetaken = {}
    sample_count = {}
    ad_dict = {}
    for ads in tqdm(bubly_ads):
        df = t[t['ad_id'] == ads]
        print("dataframe size %d for ad %d" % (len(df), ads))
        iterations = 1000000
        prior_a = 1
        prior_b = 1
        fq_dict = {}
        start_time = datetime.datetime.now()
        for i in tqdm(range(iterations)):
            sample = beta.rvs(df['clicks']+prior_a, df['non_clicks']+prior_b)
            picked_ad = int(np.argmax(sample))
            # calculating frequency of metadataid been picked for every impression
            fq_dict[picked_ad] = fq_dict.get(picked_ad,0)+1
        ad_dict[ads] = fq_dict
        finish_time = datetime.datetime.now()
        duration_sec = (finish_time - start_time).total_seconds()
        timetaken[ads] = duration_sec
        sample_count[ads] = len(fq_dict.keys())

    output = r'D:\project\cac\mab'
    with open(os.path.join('mab_dia.txt'),'w')  as f:
        #print(ad_dict)
        json.dump(ad_dict, f)

def mab_imps_simulator_v2(ds_agg_path, ds_agg_time_path, ad_id, ds_rowid_dict):
    ds_agg_data = pd.read_csv(ds_agg_path)
    ds_agg_time_data = pd.read_csv(ds_agg_time_path)
    ds_agg_data.rename(columns = {'dsrow_id':'rowid', 'ad_id':'adid', 'imp': 'imps'}, inplace = True)
    ds_agg_data['non_clicks'] = ds_agg_data['imps'] - ds_agg_data['clicks']
    ds_agg_time_data.rename(columns = {'dsrow_id':'rowid', 'clicks': 'time_clicks', 'imps':'time_imps'}, inplace = True)
    main_df = pd.DataFrame()
    for ad in ad_id:
        mds_fil = ds_agg_data[(ds_agg_data['date']=='2021-06-15') & (ds_agg_data['adid'] == ad)][['adid','rowid','imps', 'clicks','non_clicks' ]]
        mds_fil = mds_fil[mds_fil['rowid'].isin(ds_rowid_dict[str(ad)])].sort_values(['rowid'])
        mds_fil['ctr'] = (mds_fil['clicks']/mds_fil['imps'])*100
        ad_fil = ds_agg_time_data[(ds_agg_time_data['adid'] == ad) & (ds_agg_time_data['rowid'].isin(ds_rowid_dict[str(ad)]))]

        temp = pd.merge(mds_fil, ad_fil[['time_clicks', 'time_imps', 'rowid']], on = 'rowid', how = 'inner')
        served_ad_imps_count = get_ad_fq(get_served_ad_list(temp))

        fq_df = pd.DataFrame(list(served_ad_imps_count.items()), index=range(len(served_ad_imps_count))).set_index(0)
        fq_df.rename(columns = {1:'simulated_imps'}, inplace = True)
        #pd.merge(temp, fq_df, left_index=True, right_index=True)
        ad_df = temp.join(fq_df).fillna(0)
        ad_df['time_imps_dis'] = (ad_df['time_imps'] / ad_df['time_imps'].sum()) * 100
        ad_df['simulated_imps_dis'] = (ad_df['simulated_imps'] / ad_df['simulated_imps'].sum()) * 100
        plot_df = pd.melt(ad_df[['adid','rowid', 'time_imps_dis', 'simulated_imps_dis']], id_vars=["adid",'rowid'], var_name="imps_distribution", value_name="percentage")
        ad_plot = sns.catplot(x='rowid', y='percentage', hue='imps_distribution', data=plot_df, kind='bar')
        plot_name = f'adid_{ad}.png'
        ad_plot.savefig(plot_name)
        main_df = main_df.append(ad_df)



if __name__ == '__main__':
    input_path = 'D:/project/cac/mab'
    # mds_agg_data_path = os.path.join(input_path, 'metadata_ds_combined.csv')
    # mab_imps_simulator_v1(mds_agg_data_path)

    ds_agg_path = os.path.join(input_path, 'main_ds_combined.csv')
    ds_agg_time_path = os.path.join(input_path, 'dsid_6_11.csv')
    # ad id and main rowid to be tested
    ad_id = [6376, 6377, 6378]
    ds_rowid_dict = {'6376': [10905845, 10905846, 10905847, 10905848, 10932961, 10932962],
                     '6377': [10905867, 10905868, 10905869, 10905870, 10932958, 10932959],
                     '6378': [10905879, 10905880, 10905881, 10905882, 10932955, 10932956]}

    mab_imps_simulator_v2(ds_agg_path, ds_agg_time_path, ad_id, ds_rowid_dict)



