import ujson
import os
from random import shuffle


with open('../embedding/icd_mimic4_10.json', 'r') as f:
    df = ujson.load(f)
    
new_df = {}
for key, value in df.items():
    shuffle(value)
    shuffle(value)
    new_df[key] = value
    
with open('../embedding/icd_mimic4_10_random_sort.json', 'w') as f:
    ujson.dump(new_df, f, indent=2)
