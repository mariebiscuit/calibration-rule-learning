import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import xicorpy

from preprocess import process_human_data_file_to_df, process_response_file

# def calculate_human_stats(use_hdf_cache: str=None):

#     if use_hdf_cache is None:
#         hdf = process_response_file("./data/TurkData-Accuracy.txt")
#     else:
#         hdf = pd.read_csv(use_hdf_cache)

#     hdf['score'] = (hdf['answer'] == hdf['response']).astype(int)
#     # === Get human baselines
#     human_df = pd.read_csv('/users/aloo1/thesis/rq1_success/human_results/human_all.csv', index_col=0)
#     human_df['score'] = human_df['score'].astype(int)
#     # human_df = human_df.loc[:, ['id', 'concept', 'listnum', 'item_num','score']].groupby(['id', 'concept', 'listnum', 'item_num'])
#     concepts = sorted_concepts

#     stats_df = dict(concept=[], listnum=[], first_quarter=[], last_quarter=[], id=[], overall=[])

#     for concept in concepts:
#         concept_df = human_df[human_df['concept'] == concept]
#         for listnum in concept_df['listnum'].unique():
#             concept_list_df = concept_df[concept_df['listnum'] == listnum]
#             total_items = max(concept_list_df['item_num'])

#             for id in concept_list_df['id'].unique():
#                 id_concept_list_df = concept_list_df[concept_list_df['id'] == id]
#                 overall = id_concept_list_df['score'].mean()
#                 firstquarter = id_concept_list_df[id_concept_list_df['item_num'] <= total_items/4]['score'].mean()
#                 lastquarter = id_concept_list_df[id_concept_list_df['item_num'] >= (3 * (total_items/4))]['score'].mean()
#                 # firstquarter = id_concept_list_df[id_concept_list_df['item_num'] <= total_items/3]['score'].mean()
#                 # lastquarter = id_concept_list_df[id_concept_list_df['item_num'] >= (2 * (total_items/3))]['score'].mean()

#                 stats_df['overall'].append(overall)
#                 stats_df['first_quarter'].append(firstquarter)
#                 stats_df['last_quarter'].append(lastquarter)
#                 stats_df['id'].append(id)
#                 stats_df['listnum'].append(listnum)
#                 stats_df['concept'].append(concept)

#     pd.DataFrame.from_dict(stats_df).to_csv('/users/aloo1/thesis/rq1_success/human_results/human_all_quarter_stats.csv')
        


def compile_raw_results(filepath_template: str) -> pd.DataFrame:
    """
    Compiles model inference response CSVs returend from `local_inference.py` 
    into one CSV that contains scores and correlation coefficients for all concepts

    :param str file_template: filepath template with `concept` field for filenames of CSV to compile, e.g.
        ```
        './results/raw_results/gemma2b-pretrained/gemma2b-pretrained_{concept}'
        ```
    :returns:
        - DataFrame containing compiled results
    """
    hdf = process_human_data_file_to_df("./data/data.txt")
    concepts = hdf['concepts'].unique()

    df_dict = {k: [] for k in ['pyes_corr', 'pyes_corr_p', 'pyes_alpha_corr',
                                'pcorrect_corr', 'pcorrect_corr_p',
                                'spearman', 'spearman_p', 
                                'tau', 'tau_p', 
                                'xi', 'xi_p', 
                                'hscores', 'mscores',
                                'hyes', 'myes',
                                'hacc', 'macc',
                                'concept']}

    for i, concept in enumerate(concepts):
        
        hyes, hno = hdf[hdf['concepts'] == concept]['hyes'].astype(int), hdf[hdf['concepts'] == concept]['hno'].astype(int)
        htotals = hyes + hno
        hyes, hno = hyes / htotals, hno / htotals

        model_df = pd.read_csv(filepath_template.format(concept=concept))
        myes, mno = model_df['norm_true_mass'], np.ones(model_df['norm_true_mass'].shape)-model_df['norm_true_mass']

        answer_idx = (np.array(hdf[hdf['concepts'] == concept]['answers'])).astype(bool).astype(int)[:len(myes)]
        hscore = np.vstack((hno, hyes))[answer_idx, np.arange(len(hyes))][:len(myes)]

        mscore = np.vstack((mno, myes))[answer_idx, np.arange(len(myes))]

        df_dict['concept'].append(concept)
        df_dict['mscores'].append(mscore)
        df_dict['hscores'].append(hscore)
        df_dict['hyes'].append(hyes.tolist())
        df_dict['myes'].append(myes.tolist())

        pyes_alpha_corr, pyes_alpha_corr_p = pearsonr(hyes.tolist(), myes.tolist())
        pyes_corr, pyes_corr_p = pearsonr(hyes.tolist(), myes.tolist())
        pcorrect_corr, pcorrect_corr_p = pearsonr(hscore.tolist(), mscore.tolist())
        spearman, spearman_p = spearmanr(hyes.tolist(), myes.tolist())
        tau, tau_p = kendalltau(hyes.tolist(), myes.tolist())
        xi, xi_p = xicorpy.compute_xi_correlation(hyes.tolist(), myes.tolist(), get_p_values=True)

        df_dict['hacc'].append(np.mean(hscore))
        df_dict['macc'].append(np.mean(mscore))
        df_dict['xi'].append(xi)
        df_dict['xi_p'].append(xi_p)
        df_dict['spearman'].append(spearman)
        df_dict['spearman_p'].append(spearman_p)
        df_dict['tau'].append(tau)
        df_dict['tau_p'].append(tau_p)
        df_dict['pyes_alpha_corr'].append(pyes_alpha_corr)
        df_dict['pyes_corr'].append(pyes_corr)
        df_dict['pyes_corr_p'].append(pyes_corr_p)
        df_dict['pcorrect_corr'].append(pcorrect_corr)
        df_dict['pcorrect_corr_p'].append(pcorrect_corr_p)

    [print(k, len(v)) for k,v in df_dict.items()]
    df = pd.DataFrame.from_dict(df_dict)
    return df


if __name__ == "__main__":

    prefix = "gemma2b-pretrained"
    df = compile_raw_results(f"./results/raw_results/{prefix}/{prefix}" + "_{concept}.csv")
    df.to_csv(f'./results/compiled_{prefix}.csv')

    # FILE_TEMPLATE = "/users/aloo1/thesis/rq2_fit/raw_results_gemma_kl_fulldist_primitives/gemma_kl_fulldist_primitives_{concept}_L2.csv"
    # SAVETO = '/users/aloo1/thesis/rq2_fit/corr_results_kl_fulldist_primitives.csv'