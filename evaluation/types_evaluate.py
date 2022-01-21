import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import paired_distances

from greedy_evaluate import greedy_alignment

def get_types_metrics(df, embeds):
    top_k = [1, 5, 10]
    COLUMNS = ['type', 'number', 'hits1', 'hits5', 'hits10', 'mr', 'mrr']
    metrcis = pd.DataFrame(columns=COLUMNS)
    types = df['type'].unique()

    for type_ in types:
        hits = [0] * len(top_k)
        mr, mrr = 0, 0

        df_type = df.loc[df['type'] == type_]
        number_ents = len(df_type)
        df_type_en = df_type.loc[df_type['lang'] == 'en']
        ref_ent1 = list(df_type_en['ent1_id'])
        ref_ent2 = list(df_type_en['ent2_id'])
        embed1 = embeds[ref_ent1, ]
        embed2 = embeds[ref_ent2, ]

        hits, mr, mrr, cost = greedy_alignment(embed1, embed2, top_k)
        print("{}: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s".format(type_, top_k, hits, mr, mrr, cost))
        metrcis.loc[len(metrcis)] = [type_, number_ents, hits[0], hits[1], hits[2], mr, mrr]

    metrcis_sorted = metrcis.sort_values(by=['hits1'], ascending=True)
    return metrcis_sorted


def print_metrics(df, df_results, fname):
    with pd.ExcelWriter(fname) as writer:
        types = df_results['type'].unique()
        df_results.to_excel(writer, sheet_name='Types', index=False)
        for type_ in types:
            df_type = df.loc[df['type'] == type_]
            df_type_en = df_type[df_type['lang'] == 'en']

            df_pairs = pd.DataFrame(columns=['ent1', 'ent2', 'distance'])
            df_pairs['ent1'] = list(df_type_en['ent1'])
            df_pairs['ent2'] = list(df_type_en['ent2'])

            embeds = df[['x', 'y']]
            embeds = embeds.to_numpy()
            ref_ent1 = list(df_type_en['ent1_id'])
            ref_ent2 = list(df_type_en['ent2_id'])
            embed1 = embeds[ref_ent1, ]
            embed2 = embeds[ref_ent2, ]
            distances = paired_distances(embed1, embed2)
            df_pairs['distance'] = distances

            df_pairs = df_pairs.sort_values(by=['distance'], ascending=True)
            df_pairs.to_excel(writer, sheet_name=type_, index=False)


def types_evaluate(results, prepared_data, filename):
    df = pd.read_csv(prepared_data)
    ent_embeds = np.load(results + "ent_embeds.npy")

    print('Types evaluation')
    metrics = get_types_metrics(df, ent_embeds)

    print('Saving pairs')
    print_metrics(df, metrics, filename)
    
    
if __name__ == '__main__':
    results_folder = 'C:\\my-data\\output\\multike\\20210809104150\\'  # MultiKE results #Word2Vec EN-RU
    prep_data = '..\\..\\interactive-visualization\\data\\MultiKE_Word2Vec_EN_RU.csv'
    fname = 'MultiKE_Word2Vec_EN_RU.xlsx'

    fpath = '..\\data\\' + fname
    types_evaluate(results_folder, prep_data, fpath)