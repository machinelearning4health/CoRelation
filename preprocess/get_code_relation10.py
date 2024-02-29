import pickle
import csv
from itertools import combinations, product
import pickle
from tqdm import tqdm
from data_util import load_full_codes
pathmimic = 'mimic4_10'
import numpy as np
import pandas as pd
from constant import DATA_DIR
import simple_icd_10 as icd
from data_util import reformat10
dig_set = set()
with open("%s/d_icd10_diagnoses.csv" % (DATA_DIR), 'r') as descfile:
    r = csv.reader(descfile)
    # header
    next(r)
    for row in r:
        if row[1] == '10':
            code = row[0]
            desc = row[-1]
            dig_set.add(code.replace('.', ''))
pro_set = set()
with open("%s/d_icd10_procedures.csv" % (DATA_DIR), 'r') as descfile:
    r = csv.reader(descfile)
    # header
    next(r)
    for row in r:
        if row[1] == '10':
            code = row[0]
            desc = row[-1]
            pro_set.add(code.replace('.', ''))

G_node = None
def get_path(a):
    path = set()
    while a is not G_node:
        path.add(a.code)
        a = a.parent
    return path

def get_relation_distance(a,b):
    a_path = set(a)
    b_path = set(b)
    distance = len(a_path.symmetric_difference(b_path))
    return distance

def proc_list(code):
    list_ans = []
    count = 1
    while count < len(code):
        list_ans.append(code[:count])
        count += 1
    return list_ans


ind2c, desc_dict = load_full_codes('mimicdata/{}/train_full.csv'.format(pathmimic), version='mimic4_10')
ind2c_50, desc_dict_50 = load_full_codes('mimicdata/{}/train_50.csv'.format(pathmimic), version='mimic4_10')
distances_50 = {}
for code in tqdm(ind2c_50.values()):
    distances_50[(code, code)] = 0
major_codes = []
for x in ind2c.values():
    if icd.is_valid_item(x):
        major_codes.append(x[0:3])
major_codes = list(set(major_codes))
for relation in tqdm(product(ind2c_50.values(), major_codes)):
    a_ = reformat10(relation[0], relation[0] in dig_set)
    a = icd.get_ancestors(a_) if icd.is_valid_item(a_) else proc_list(a_)
    b = icd.get_ancestors(relation[1]) if icd.is_valid_item(relation[1]) else proc_list(relation[1])
    distance = get_relation_distance(a, b) if icd.is_valid_item(a_) == icd.is_valid_item(relation[1]) else 10
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
    a_ = reformat10(relation[0], relation[0] in dig_set)
    a = icd.get_ancestors(a_) if icd.is_valid_item(a_) else proc_list(a_)
    b = icd.get_ancestors(relation[1]) if icd.is_valid_item(relation[1]) else proc_list(relation[1])
    distance = get_relation_distance(a, b) if icd.is_valid_item(a_) == icd.is_valid_item(relation[1]) else 10
    if distance > 10:
        label = 10
    else:
        label = distance
    #distances[relation] = 1
    distances[(relation[1], relation[0])] = label

pickle.dump(distances, open('mimicdata/{}/full_relation.pkl'.format(pathmimic), 'wb'))