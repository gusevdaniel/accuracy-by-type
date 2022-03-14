import numpy as np
import pandas as pd

def delete_host(uri):
    uri = uri.strip('\n').split('/')
    return uri[-1]


def loadIds(fn, inverse=False):
    pair = dict()
    with open(fn, encoding='utf-8') as f:
        if inverse:
            for line in f:
                th = line[:-1].split('\t')
                pair[th[0]] = int(th[1])
        else:
            for line in f:
                th = line[:-1].split('\t')
                pair[int(th[1])] = th[0]
    return pair


def get_kgs_ids(folderpath, inverse=False):
    kg_ids_1 = loadIds(folderpath + 'kg1_ent_ids', inverse)
    kg_ids_2 = loadIds(folderpath + 'kg2_ent_ids', inverse)

    kgs_ids = dict()
    kgs_ids.update(kg_ids_1)
    kgs_ids.update(kg_ids_2)
    return kgs_ids