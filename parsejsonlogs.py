from tqdm import tqdm
import pickle
import numpy as np
import fnmatch
import os
import json
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import collections


DIR = '/root/azure-storage/eval_outputs/mmlu/jsonlogs/wanda'
models = ['llama13b']
mmlu_tasks=['hendrycksTest-abstract_algebra', 'hendrycksTest-anatomy', 'hendrycksTest-astronomy', 'hendrycksTest-business_ethics', 'hendrycksTest-clinical_knowledge', 'hendrycksTest-college_biology', 'hendrycksTest-college_chemistry', 'hendrycksTest-college_computer_science', 'hendrycksTest-college_mathematics', 'hendrycksTest-college_medicine', 'hendrycksTest-college_physics', 'hendrycksTest-computer_security', 'hendrycksTest-conceptual_physics', 'hendrycksTest-econometrics', 'hendrycksTest-electrical_engineering', 'hendrycksTest-elementary_mathematics', 'hendrycksTest-formal_logic', 'hendrycksTest-global_facts', 'hendrycksTest-high_school_biology', 'hendrycksTest-high_school_chemistry', 'hendrycksTest-high_school_computer_science', 'hendrycksTest-high_school_european_history', 'hendrycksTest-high_school_geography', 'hendrycksTest-high_school_government_and_politics', 'hendrycksTest-high_school_macroeconomics', 'hendrycksTest-high_school_mathematics', 'hendrycksTest-high_school_microeconomics', 'hendrycksTest-high_school_physics', 'hendrycksTest-high_school_psychology',	'hendrycksTest-high_school_statistics', 'hendrycksTest-high_school_us_history', 'hendrycksTest-high_school_world_history', 'hendrycksTest-human_aging', 'hendrycksTest-human_sexuality', 'hendrycksTest-international_law', 'hendrycksTest-jurisprudence', 'hendrycksTest-logical_fallacies', 'hendrycksTest-machine_learning', 'hendrycksTest-management', 'hendrycksTest-marketing', 'hendrycksTest-medical_genetics', 'hendrycksTest-miscellaneous', 'hendrycksTest-moral_disputes', 'hendrycksTest-moral_scenarios', 'hendrycksTest-nutrition', 'hendrycksTest-philosophy', 'hendrycksTest-prehistory', 'hendrycksTest-professional_accounting', 'hendrycksTest-professional_law', 'hendrycksTest-professional_medicine', 'hendrycksTest-professional_psychology', 'hendrycksTest-public_relations', 'hendrycksTest-security_studies', 'hendrycksTest-sociology', 'hendrycksTest-us_foreign_policy', 'hendrycksTest-virology', 'hendrycksTest-world_religions']

shots = ['fewshot5']
sps = [0.025, 0.05, 0.075, 0.125, 0.15, 0.175, 0.1, 0.225, 0.25, 0.275, 0.2, 0.325, 0.35, 0.375, 0.3, 0.425, 0.45, 0.475, 0.4, 0.5]

def getinfo(fname):
    model, sp , shot , task = None, None, None, None
    for m in models:
        if m in fname:
            model = m
            break
    for t in mmlu_tasks:
        if t in fname:
            task = t
            break
    sp = (f.split('sp')[1])
    sp = sp[0] + '.' + sp[1:]
    sp = float(sp)

    for sh in shots:
        if sh in fname:
            shot = sh
            break  
    
    if model is None or sp is None or shot is None or task is None:
        print('Error in parsing file: ', fname)
        exit()
    return model, sp , shot , task


files = os.listdir(DIR)
dict_f_content = {}
for f in tqdm(files):
    model, sp , shot , task = getinfo(f)
    dict_f_content[(model, sp, shot, task)] = json.load(open(os.path.join(DIR, f)))

for model in models:
    for sp in sps:
        for shot in shots:
            for task in mmlu_tasks:
                if (model, sp, shot, task) not in dict_f_content:
                    print('Missing: ', model, sp, shot, task)
                    

breakpoint()


