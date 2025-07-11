# Code to fetch the score array for a given state, chamber, ensemble_type and score.
# and also creates the lists of ensemble types, score, and state-chamber combinations.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import json
from itertools import combinations

local_folder = 'C:/Users/ktapp/Documents/Python/vanilla ensembles'

state_list = ['FL', 'IL', 'MI', 'NC', 'NY', 'OH', 'WI'] 
state_chamber_list = [(state, chamber) for state in state_list for chamber in ['congress', 'upper', 'lower']]

base_list = ['base0', 'base1', 'base2', 'base3', 'base4']
ensemble_list = ['pop_minus', 'pop_plus', 'distpair','ust', 'distpair_ust', 'reversible', 'county25', 'county50', 'county75', 'county100']

#convert into LaTex notation in our paper
ensemble_name_dict = {
'base0': '$\RA_0$',
'base1': '$\RA_1$',
'base2': '$\RA_2$',
'base3': '$\RA_3$',
'base4': '$\RA_4$',
'pop_minus': '$\popm$',
'pop_plus': '$\popp$',
'ust': '$\RC$',
'distpair': '$\RB$',
'distpair_ust': '$\RD$',
'reversible': '\RevReCom',
'county25': '$\C$',
'county50': '$\CC$',
'county75':'$\CCC$',
'county100': '$\CCCC$'  
}
reverse_ensemble_name_dict = {v: k for k, v in ensemble_name_dict.items()}

num_seats_dict = {
    ('FL', 'congress'): 28,
    ('FL', 'upper'): 40,
    ('FL', 'lower'): 120,
    ('IL', 'congress'): 17,
    ('IL', 'upper'): 59,
    ('IL', 'lower'): 118,
    ('MI', 'congress'): 13,
    ('MI', 'upper'): 38,
    ('MI', 'lower'): 110,
    ('NC', 'congress'): 14,
    ('NC', 'upper'): 50,
    ('NC', 'lower'): 120,
    ('NY', 'congress'): 26,
    ('NY', 'upper'): 63,
    ('NY', 'lower'): 150,
    ('OH', 'congress'): 15,
    ('OH', 'upper'): 33,
    ('OH', 'lower'): 99,
    ('WI', 'congress'): 8,
    ('WI', 'upper'): 33,
    ('WI', 'lower'): 99
    }

with open('score_categories.json', 'r') as file: # Read in Alec's dictionary of caterorized scores.
    score_categories = json.load(file)

# create dictionary mapping my version of each primary score name to the corresponding name in Alec's dictionary
primary_score_dict = {
    'Reock': "reock",
    'Polsby-Popper': "polsby_popper",
    'cut edges': "cut_score", 
    'Dem seats': "fptp_seats",
    'efficiency gap': "efficiency_gap_wasted_votes",
    'mean-median': "mean_median_average_district",
    'partisan bias': "geometric_seats_bias", 
    'competitive districts':  "competitive_district_count",
    'average margin': "average_margin", 
    'MMD black': "mmd_black",
    'MMD hispanic': "mmd_hispanic",
    'MMD coalition': "mmd_coalition",
    'county splits':"county_splits", 
    'counties split':"counties_split"
}

# List of primary scores and list of secondary scores
primary_score_list = list(primary_score_dict.keys())
secondary_score_list = [score for ls in score_categories.values() for score in ls if score not in primary_score_dict.values()]

# dictionary mapping each score from primary_score_list and secondary_score_list to info about the spreadsheet column where it is stored
score_to_spreadsheet_info = {}
for category, scores in score_categories.items():
    for score in scores:
        score_to_spreadsheet_info[score] = (f'{category}_scores.csv', score)

for my_score_name, score in primary_score_dict.items():
    score_to_spreadsheet_info[my_score_name] = score_to_spreadsheet_info[score]

def fetch_score_array(state, chamber, ensemble_type, score):
    """
    Fetches the score array for the given state, chamber and ensemble_type.
    If score is 'reock', 'polsby_popper', etc.,
    then it returns 1D array containing the scores of the maps in the ensemble.
    If score == 'by_district', then it returns a 2D array containing,
    for each map in the ensemble, an ordered array recording the dem_portions of the districts of the map.
    """

    if chamber == 'congress':
        pop0 = '0.01'
        pop_minus = '0.005'
        pop_plus = '0.015'
    else:
        pop0 = '0.05'
        pop_minus = '0.025'
        pop_plus = '0.075'

    type0 = 'cut-edges-rmst'
    type1 = 'cut-edges-region-aware'
    county0 = '0.0'

    ensemble_dict = {
        'base0' : f'T{pop0}_S{county0}_R0_V{type0}',
        'base1' : f'T{pop0}_S{county0}_R1_V{type0}',
        'base2' : f'T{pop0}_S{county0}_R2_V{type0}',
        'base3' : f'T{pop0}_S{county0}_R3_V{type0}',
        'base4' : f'T{pop0}_S{county0}_R4_V{type0}',
        'pop_minus' : f'T{pop_minus}_S{county0}_R0_V{type0}',
        'pop_plus' : f'T{pop_plus}_S{county0}_R0_V{type0}',
        'ust' : f'T{pop0}_S{county0}_R0_Vcut-edges-ust',
        'distpair' : f'T{pop0}_S{county0}_R0_Vdistrict-pairs-rmst',
        'distpair_ust' : f'T{pop0}_S{county0}_R0_Vdistrict-pairs-ust',
        'reversible' : f'T{pop0}_S{county0}_R0_Vreversible',
        'county25' : f'T{pop0}_S{0.25}_R0_V{type1}',
        'county50' : f'T{pop0}_S{0.5}_R0_V{type1}',
        'county75' : f'T{pop0}_S{0.75}_R0_V{type1}',
        'county100' : f'T{pop0}_S{1.0}_R0_V{type1}',   
    }
    snipet = ensemble_dict[ensemble_type]

    score_sheet, col_name = score_to_spreadsheet_info[score] if score != 'by_district' else ('partisan_bydistrict.jsonl', None)
    filename = f'{local_folder}/{state}_{chamber}/{state}_{chamber}/{state}_{chamber}_{snipet}/{state}_{chamber}_{snipet}_{score_sheet}'
    if score == 'by_district':
        with open(filename,'r', encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
        to_return = []
        for i in range(1, len(data)):
            dem_counts = data[i]['by-district']['election']['C16GCO']['dem_by_district']
            tot_counts = data[i]['by-district']['election']['C16GCO']['tot_by_district']
            dem_portions = [dem_counts[i] / tot_counts[i] for i in range(1,len(dem_counts))]
            dem_portions.sort()
            to_return.append(np.array(dem_portions))
        return np.array(to_return)
    else:
        df = pd.read_csv(filename)
        return df[col_name].to_numpy()