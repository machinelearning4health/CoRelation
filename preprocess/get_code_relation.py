import pickle
import csv
from itertools import combinations, product
import pickle
from tqdm import tqdm
from data_util import load_full_codes
pathmimic = 'mimic4_9'
import numpy as np
import pandas as pd

from icd9cms.icd9 import search
class SGCode(object):
    def __init__(self, code, text):
        self.code = code
        self.long_desc = text
        self.parent = None
        self.children = []
ICD9SG = pickle.load(open('../mimicdata/ICD9SG.pkl', 'rb'))

def searchSG(code):
    code = code.replace('.', '')
    if code in ICD9SG:
        return ICD9SG[code]


def get_children(a):
    temp = []
    temp += [a.code]
    if a.children is not None:
        for child in a.children:
            temp += get_children(child)
    return temp

def propogate_exclusion(exclusion_map):
    for relation in exclusion_map:
        a = search(relation[0]) if search(relation[0]) is not None else searchSG(relation[0])
        b = search(relation[1]) if search(relation[1]) is not None else searchSG(relation[1])
        if a is None or b is None:
            continue
        a_sons = get_children(a)
        b_sons = get_children(b)
        for ab_group in product(a_sons, b_sons):
            if ab_group not in exclusion_map:
                exclusion_map.add(ab_group)
    return exclusion_map
G_node = None
def get_path(a):
    path = set()
    while a is not G_node:
        path.add(a.code)
        a = a.parent
    return path

def get_relation_distance(a,b):
    a_path = get_path(a)
    b_path = get_path(b)
    distance = len(a_path.symmetric_difference(b_path))
    return distance


ind2c, desc_dict = load_full_codes('mimicdata/{}/train_full.csv'.format(pathmimic), version='mimic3')
ind2c_50, desc_dict_50 = load_full_codes('mimicdata/{}/train_50.csv'.format(pathmimic), version='mimic3-50')
distances_50 = {}
for code in tqdm(ind2c_50.values()):
    distances_50[(code, code)] = 0
major_codes = list(set([x.split('.')[0] for x in ind2c.values()]))
for relation in tqdm(product(ind2c_50.values(), major_codes)):
    a = search(relation[0]) if search(relation[0]) is not None and search(relation[0]).alt_code == relation[0] else searchSG(relation[0])
    b = search(relation[1]) if search(relation[1]) is not None and search(relation[1]).alt_code == relation[1] else searchSG(relation[1])
    distance = get_relation_distance(a, b) if type(a) == type(b) else 10
    if distance > 10:
        label = 10
    else:
        label = distance
    #distances_50[relation] = 1
    distances_50[(relation[1], relation[0])] = label
pickle.dump(distances_50, open('mimicdata/{}/50_relation.pkl'.format(pathmimic), 'wb'))


distances = {}
for code in tqdm(ind2c.values()):
    distances[(code, code)] = 0
for relation in tqdm(product(ind2c.values(), major_codes)):
    a = search(relation[0]) if search(relation[0]) is not None and search(relation[0]).alt_code == relation[0] else searchSG(relation[0])
    b = search(relation[1]) if search(relation[1]) is not None and search(relation[1]).alt_code == relation[1] else searchSG(relation[1])
    distance = get_relation_distance(a, b) if type(a) == type(b) else 10
    if distance > 10:
        label = 10
    else:
        label = distance
    #distances[relation] = 1
    distances[(relation[1], relation[0])] = label

pickle.dump(distances, open('mimicdata/{}/full_relation.pkl'.format(pathmimic), 'wb'))