import sys
maxInt = sys.maxsize
import csv
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
sys.path.append('../')
# put mimic3 related files in MIMIC_3_DIR
from constant import MIMIC_3_DIR
import numpy as np
import pandas as pd
from icd_50 import codes_50
from tqdm import tqdm
import csv
note_dict = {}
others_dict = {}

notes_file = '%s/NOTEEVENTS.csv' % (MIMIC_3_DIR)
with open(notes_file, 'r') as csvfile:
    notereader = csv.reader(csvfile)
    next(notereader)
    i = 0
    for line in tqdm(notereader):
        subject_id = int(line[1])
        hadm_id = str(line[2])
        category = str(line[6])
        note = line[10]
        if (subject_id, hadm_id) not in note_dict:
            note_dict[(subject_id, hadm_id)] = []
            others_dict[(subject_id, hadm_id)] = []
        if category == "Discharge summary":
            note_dict[(subject_id, hadm_id)].append(note)
        else:
            others_dict[(subject_id, hadm_id)].append(category + ": " + note)
        i += 1
df_codes = pd.read_csv('%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR, index_col=None)
print(len(set(df_codes['HADM_ID'])))

def split_note_to_paragraph(note):
    result = {}
    now_paragraph = 'other'
    for line in note.split('\n'):
        if line.find(":") >= 0:
            now_paragraph = line[0:line.find(":")]
            text = line[line.find(":") + 1:]
            if not now_paragraph in result:
                result[now_paragraph] = []
            result[now_paragraph].append(text.strip())
        elif not line.strip():
            now_paragraph = 'other'
        else:
            if not now_paragraph in result:
                result[now_paragraph] = []
            result[now_paragraph].append(line.strip())
    for paragraph in result:
        result[paragraph] = " ".join(result[paragraph])
    return result
from tqdm import trange

icd_dict = {}
for i in trange(df_codes.shape[0]):
    subject_id = df_codes['SUBJECT_ID'][i]
    hadm_id = str(df_codes['HADM_ID'][i])
    code = df_codes['ICD9_CODE'][i]
    if (subject_id, hadm_id) not in icd_dict:
        icd_dict[(subject_id, hadm_id)] = []
    icd_dict[(subject_id, hadm_id)].append(code)

for base_name in ['mimic3', 'mimic3-50']:
    train_name = '%s_train.json' % (base_name)
    dev_name = '%s_dev.json' % (base_name)
    test_name = '%s_test.json' % (base_name)

    hadm_ids = {}

    #read in train, dev, test splits
    for splt in ['train', 'dev', 'test']:
        hadm_ids[splt] = set()
        if base_name == "mimic3":
            base = "full"
        if base_name == "mimic3-50":
            base = "50"
        with open('%s/%s_%s_hadm_ids.csv' % (MIMIC_3_DIR, splt, base), 'r') as f:
            for line in f:
                hadm_ids[splt].add(line.rstrip())

    train_list = []
    dev_list = []
    test_list = []

    for key in tqdm(icd_dict.keys()):
        subject_id = str(int(key[0]))
        hadm_id = str(int(key[1]))
        icd = ";".join([str(c) for c in icd_dict[key]])
        if base_name == "mimic-50":
            filtered_codes = set(icd_dict[key]).intersection(set(codes_50))
            if len(filtered_codes) > 0:
                icd = ";".join([str(c) for c in filtered_codes])
            else:
                continue

        text = "\t".join([str(c) for c in note_dict[key]])
        additional_text = "\t".join([str(c) for c in others_dict[key]])
        row = {'subject_id':subject_id,
               'hadm_id':hadm_id,
               'LABELS':icd,
               'TEXT':text,
               'Addition':additional_text}

        if hadm_id in hadm_ids['train']:
            train_list.append(row)
        elif hadm_id in hadm_ids['dev']:
            dev_list.append(row)
        elif hadm_id in hadm_ids['test']:
            test_list.append(row)
        else:
            #print(key)
            pass

    print(len(train_list), len(dev_list), len(test_list))
    import json
    with open(train_name, "w") as f:
        json.dump(train_list, f, indent=4)
    with open(dev_name, "w") as f:
        json.dump(dev_list, f, indent=4)
    with open(test_name, "w") as f:
        json.dump(test_list, f, indent=4)