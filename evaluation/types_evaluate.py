import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics.pairwise import paired_distances

from greedy_evaluate import greedy_alignment

from utils import *


def get_overall_accuracy(df, embeds):
    top_k = [1, 5, 10, 50]
    overall = pd.DataFrame(columns=['hits1', 'hits5', 'hits10', 'hits50', 'mr', 'mrr'])

    hits = [0] * len(top_k)
    mr, mrr = 0, 0

    df_en = df.loc[df['lang'] == 'en']
    ref_ent1 = list(df_en['ent1_id'])
    ref_ent2 = list(df_en['ent2_id'])
    embed1 = embeds[ref_ent1, ]
    embed2 = embeds[ref_ent2, ]

    hits, mr, mrr, cost = greedy_alignment(embed1, embed2, top_k)
    print("{}: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s".format('Overall', top_k, hits, mr, mrr, cost))
    overall.loc[len(overall)] = [hits[0], hits[1], hits[2], hits[3], mr, mrr]
    return overall



def get_types_metrics(df, embeds):
    top_k = [1, 5, 10]
    COLUMNS = ['type', 'number', 'hits1', 'hits5', 'hits10', 'mr', 'mrr']
    metrcis = pd.DataFrame(columns=COLUMNS)
    types = df['type'].unique()

    for type_ in tqdm(types):
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
        #print("{}: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s".format(type_, top_k, hits, mr, mrr, cost))
        metrcis.loc[len(metrcis)] = [type_, number_ents, hits[0], hits[1], hits[2], mr, mrr]

    metrcis_sorted = metrcis.sort_values(by=['hits1'], ascending=True)
    return metrcis_sorted


def read_relation_triples(file_path):
    print("read relation triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, relations = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 3
        h = params[0].strip()
        r = params[1].strip()
        t = params[2].strip()
        triples.add((h, r, t))
        entities.add(h)
        entities.add(t)
        relations.add(r)
    return triples, entities, relations


def read_attribute_triples(file_path):
    print("read attribute triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, attributes = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip().strip('\n').split('\t')
        if len(params) < 3:
            continue
        head = params[0].strip()
        attr = params[1].strip()
        value = params[2].strip()
        if len(params) > 3:
            for p in params[3:]:
                value = value + ' ' + p.strip()
        value = value.strip().rstrip('.').strip()
        entities.add(head)
        attributes.add(attr)
        triples.add((head, attr, value))
    return triples, entities, attributes


def count_unique_elements(triples):
    result = dict()
    for triplet in triples:
        unique = set(triplet)
        for elem in unique:
            if elem in result:
                result[elem] = result[elem] + 1
            else:
                result[elem] = 1
    return result


def form_ent_matches(kgs_ids, num_unique):
    matches = dict()
    for elem in num_unique:
        if elem in kgs_ids:
            ent_id = kgs_ids[elem]
            num = num_unique[elem]
            matches[ent_id] = num
    return matches


def print_metrics(source_folder, results_folder, df, df_overall, df_results, fname):
    kgs_ids = get_kgs_ids(results_folder, True)
    rel_triples_1, _, _ = read_relation_triples(source_folder + 'rel_triples_1')
    num_unique_rel_1 = count_unique_elements(rel_triples_1)
    ent_rel_num_1 = form_ent_matches(kgs_ids, num_unique_rel_1)

    rel_triples_2, _, _ = read_relation_triples(source_folder + 'rel_triples_2')
    num_unique_rel_2 = count_unique_elements(rel_triples_2)
    ent_rel_num_2 = form_ent_matches(kgs_ids, num_unique_rel_2)

    attr_triples_1, _, _ = read_attribute_triples(source_folder + 'attr_triples_1')
    num_unique_attr_1 = count_unique_elements(attr_triples_1)
    ent_attr_num_1 = form_ent_matches(kgs_ids, num_unique_attr_1)

    attr_triples_2, _, _ = read_attribute_triples(source_folder + 'attr_triples_2')
    num_unique_attr_2 = count_unique_elements(attr_triples_2)
    ent_attr_num_2 = form_ent_matches(kgs_ids, num_unique_attr_2)

    with pd.ExcelWriter(fname) as writer:
        df_overall.to_excel(writer, sheet_name='Overall', index=False)

        types = df_results['type'].unique()
        df_results.to_excel(writer, sheet_name='Types', index=False)
        for type_ in types:
            df_type = df.loc[df['type'] == type_]
            df_type_en = df_type[df_type['lang'] == 'en']

            df_pairs = pd.DataFrame(columns=['ent1_id', 'ent2_id', 'ent1', 'ent2', 'distance'])
            df_pairs['ent1_id'] = list(df_type_en['ent1_id'])
            df_pairs['ent2_id'] = list(df_type_en['ent2_id'])
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
            df_pairs = df_pairs.sort_values(by=['distance'], ascending=False)

            df_pairs['rel_num_1'] = df_pairs['ent1_id'].map(ent_rel_num_1)
            df_pairs['rel_num_2'] = df_pairs['ent2_id'].map(ent_rel_num_2)

            df_pairs['attr_num_1'] = df_pairs['ent1_id'].map(ent_attr_num_1)
            df_pairs['attr_num_2'] = df_pairs['ent2_id'].map(ent_attr_num_2)

            df_pairs.to_excel(writer, sheet_name=type_, index=False)


def types_evaluate(source, results, prepared_data, filename):
    df = pd.read_csv(prepared_data)
    ent_embeds = np.load(results + "ent_embeds.npy")

    print('Overall accuracy')
    overall = get_overall_accuracy(df, ent_embeds)

    print('Types evaluation')
    metrics = get_types_metrics(df, ent_embeds)

    print('Getting additional data')
    print_metrics(source, results, df, overall, metrics, filename)
    
    
if __name__ == '__main__':
    source_folder = 'C:\\my-data\\EN_RU_15K\\EN_RU_15K_V1\\'
    results_folder = 'C:\\my-data\\output\\results\\MultiKE\\EN_RU_15K_V1\\631\\20220305043056\\'
    prep_data = '..\\..\\interactive-visualization\\data\\MultiKE_EN_RU_15K_V1_swvg.csv'
    fname = 'MultiKE_EN_RU_15K_V1_swvg.xlsx'

    print(source_folder)
    print(results_folder)
    print(prep_data)
    print(fname)

    fpath = '..\\data\\' + fname
    types_evaluate(source_folder, results_folder, prep_data, fpath)