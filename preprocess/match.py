import os
import pandas as pd
import json
import unicodedata
from tqdm import trange
import sys
sys.path.append("..")
from constant import MIMIC_2_DIR, MIMIC_3_DIR, UMLS_PATH, MIMIC_4_DIR, MIMIC_4_10_DIR, DATA_DIR
from load_umls import UMLS
from data_util import load_code_descriptions, load_full_codes, reformat10
import csv


def strip_accents(text):
    return ''.join(char for char in
                   unicodedata.normalize('NFKD', text)
                   if unicodedata.category(char) != 'Mn')


def load_icd_dict(version):
    if version == "mimic2":
        cache_path = "icd_mimic2.json"
    if version in ["mimic3"]:
        cache_path = "icd_mimic3.json"
    if version in ["mimic3-50"]:
        cache_path = "icd_mimic3-50.json"
    if version in ["mimic4"]:
        cache_path = "./embedding/icd_mimic4_9.json"
    if version in ["mimic4_10"]:
        cache_path = "./embedding/icd_mimic4_10.json"
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

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            icd_dict = json.load(f)
        return icd_dict

    if version == 'mimic2':
        raise NotImplementedError
    if version == 'mimic3':
        train_path = os.path.join(MIMIC_3_DIR, "train_full.csv")
    if version == 'mimic3-50':
        train_path = os.path.join(MIMIC_3_DIR, "train_50.csv")
    if version == 'mimic4':
        train_path = os.path.join(MIMIC_4_DIR, "train_full.csv")
    if version == 'mimic4_10':
        train_path = os.path.join(MIMIC_4_10_DIR, "train_full.csv")
    ind2c, _ = load_full_codes(train_path, version=version)

    if version == 'mimic2':
        desc_dict = load_code_descriptions(version)
    elif version == 'mimic3':
        desc_dict = load_code_descriptions('mimic3')
    else:
        desc_dict = load_code_descriptions(version)

    umls = UMLS(UMLS_PATH, only_load_dict=True)

    icd_dict = {}
    desc_count = 0
    umls_count = 0
    for icd in ind2c.values():
        if version != 'mimic4_10':
            icd_ = icd
        else:
            icd_ = reformat10(icd, icd in dig_set)
        icd_dict[icd] = []
        if icd in desc_dict:
            icd_dict[icd] = [desc_dict[icd]] + umls.icd2str(icd_)
            desc_count += 1
            if umls.icd2str(icd):
                umls_count += 1
        elif icd.endswith('.'):
            if icd[0:-1] in desc_dict:
                icd_dict[icd] = [desc_dict[icd[0:-1]]] + umls.icd2str(icd[0:-1])
                desc_count += 1
                if umls.icd2str(icd[0:-1]):
                    umls_count += 1
        else:
            icd_dict[icd] = umls.icd2str(icd_)
            if umls.icd2str(icd):
                umls_count += 1
        icd_dict[icd] = list(set([strip_accents(w).lower() for w in icd_dict[icd]]))

    print(f"ICD count:{len(icd_dict)}")
    print(f"With Desc:{desc_count}")
    print(f"From UMLS:{umls_count}")

    with open(cache_path, 'w') as f:
        json.dump(icd_dict, f)

    return icd_dict


if __name__ == "__main__":
    load_icd_dict('mimic4_10')
