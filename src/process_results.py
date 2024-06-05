import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import xicorpy

from preprocess import process_human_data_file_to_df, process_response_file

"""
=== Experiment 1 and 3 ===
Data processors to generate dataframes compatible with visualization notebooks
from inference code
"""
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

        
def process_bayesian_results_to_df(file_path: str):
    """
    :param str file_path: Path to file containing bayesian model outputs that look like this
        ```
        hg13L2	0	0	0	8	16	0.854544341564178466796875	0.43748861530229987693019211292267	"λx.apply1(apply2(same_color,x.o),x.o)"
        hg13L2	1	0	0	11	13	0.854544341564178466796875	0.48052608880027491977671161293983	"λx.apply1(apply2(same_color,x.o),x.o)"
        ```
    """
    df_dict = {k: [] for k in ['pyes_corr', 'pyes_corr_p',
                                'pcorrect_corr', 'pcorrect_corr_p',
                                'spearman', 'spearman_p', 
                                'tau', 'tau_p', 
                                'xi', 'xi_p', 
                                'hscores', 'mscores', 
                                'hyes', 'myes',
                                'hacc', 'macc',
                                'concept']}

    hdf = process_human_data_file_to_df('./data/data.txt', "L2")
    concept= hdf['concepts'].unique()

    current_concept = None
    myes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            conceptlist, item, setnum, label, _, _, _, mposterior, _ = line.split("\t")
            concept, listnum = conceptlist[:-2], conceptlist[-2:]

            if listnum == "L2":
                if current_concept is None:
                    current_concept = concept

                if (current_concept != concept) or i == (len(lines) -1):
                    # flush
                    if (i == (len(lines) - 1)):
                        myes.append(float(mposterior)) 

                    myes = np.array(myes)
                    mno = 1 - myes
                    
                    hyes, hno = hdf[hdf['concepts'] == current_concept]['hyes'].astype(int), hdf[hdf['concepts'] == current_concept]['hno'].astype(int)

                    htotals = hyes + hno
                    hyes, hno = hyes / htotals, hno / htotals
                    hyes, hno = hyes[:len(myes)], hno[:len(mno)]
                    
                    if len(myes) != len(hyes):
                        print("myes", myes)

                    if isinstance(np.array(hdf[hdf['concepts'] == current_concept]['answers'])[0], str):
                        answer_idx = (np.array(hdf[hdf['concepts'] == current_concept]['answers']) == "True").astype(int)
                    elif isinstance(np.array(hdf[hdf['concepts'] == current_concept]['answers'])[0], bool):
                        answer_idx = (np.array(hdf[hdf['concepts'] == current_concept]['answers'])).astype(int)
                    else:
                        raise ValueError("DataFrame 'answers' column neither string nor boolean.")

                    hscore = np.vstack((hno, hyes))[answer_idx, np.arange(len(hyes))]
                    mscore = np.vstack((mno, myes))[answer_idx, np.arange(len(myes))]
                        
                    df_dict['concept'].append(current_concept)
                    df_dict['mscores'].append(mscore)
                    df_dict['hscores'].append(hscore)
                    df_dict['hyes'].append(hyes.tolist())
                    df_dict['myes'].append(myes.tolist())

                    pyes_corr, pyes_corr_p = pearsonr(hyes.tolist(), myes.tolist())
                    pcorrect_corr, pcorrect_corr_p = pearsonr(hscore.tolist(), mscore.tolist())
                    spearman, spearman_p = spearmanr(hscore.tolist(), mscore.tolist())
                    tau, tau_p = kendalltau(hscore.tolist(), mscore.tolist())
                    xi, xi_p = xicorpy.compute_xi_correlation(hscore.tolist(), mscore.tolist(), get_p_values=True)

                    df_dict['hacc'].append(np.mean(hscore))
                    df_dict['macc'].append(np.mean(mscore))
                    df_dict['xi'].append(xi)
                    df_dict['xi_p'].append(xi_p)
                    df_dict['spearman'].append(spearman)
                    df_dict['spearman_p'].append(spearman_p)
                    df_dict['tau'].append(tau)
                    df_dict['tau_p'].append(tau_p)
                    df_dict['pyes_corr'].append(pyes_corr)
                    df_dict['pyes_corr_p'].append(pyes_corr_p)
                    df_dict['pcorrect_corr'].append(pcorrect_corr)
                    df_dict['pcorrect_corr_p'].append(pcorrect_corr_p)

                    # reset
                    myes = []
                    myes.append(float(mposterior))
                    current_concept = concept
                else:
                    myes.append(float(mposterior)) 

    [print(k, len(v)) for k,v in df_dict.items()]
    return pd.DataFrame.from_dict(df_dict)


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

        if isinstance(np.array(hdf[hdf['concepts'] == concept]['answers'])[0], str):
            answer_idx = (np.array(hdf[hdf['concepts'] == concept]['answers']) == "True").astype(int)
        elif isinstance(np.array(hdf[hdf['concepts'] == concept]['answers'])[0], bool):
            answer_idx = (np.array(hdf[hdf['concepts'] == concept]['answers'])).astype(int)
        else:
            raise ValueError("DataFrame 'answers' column neither string nor boolean.")

        hscore = np.vstack((hno, hyes))[answer_idx, np.arange(len(hyes))]
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

    # [print(k, len(v)) for k,v in df_dict.items()] # debugging check
    df = pd.DataFrame.from_dict(df_dict)
    print("Done processing!")
    return df


if __name__ == "__main__":

    # process_bayesian_results_to_df("./data/PTG16_model.txt").to_csv('./datasets/compiled_ptg16.csv')

    prefix = [
        # "gemma2b-tuned112",
        # "gemma7b-tuned112",
        # "gemma7b-tuned92",
        # "gemma7b-tuned112-answerloss",
        # "gemma7b-tuned92-and-special",
        # 'gemma2b-pretrained',
        'gemma7b-pretrained',
        # 'gemma2b-tuned112-sparsed_primitivesor'
    ][0]

    df = compile_raw_results(f"./results/experiment_2/raw_results/{prefix}/{prefix}" + "_{concept}.csv")
    df.to_csv(f'./results/experiment_2/compiled_{prefix}.csv')