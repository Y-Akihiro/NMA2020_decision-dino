# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:24:26 2020

@author: Akihiro

Steps: 
        i) Load Data (...Complete)
        ii) Sort Data by mice (...Complete)
        iii) Analyze (make psychometric curve) by mice (...Working)
"""

import os
import functions_get_ as fun
import numpy as np
import matplotlib.pyplot as plt

# Load data (run functions in 'main_behavior_analysis.py')
fname = [] # initialize the list
fname = fun.load_names()
print(fname)

srcdir = os.getcwd()
print('Current directory is...', srcdir)

# ====================== i) Lost Stuff ======================
alldat = np.array([])
for j in range(len(fname)):
    alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))

# ====================== ii) Sort Stuff Out ======================
mouse_names = []
for n_session in range(39):
    print(alldat[n_session]['mouse_name'])
    mouse_names.append(alldat[n_session]['mouse_name'])

unique_names, name_counts = np.unique(mouse_names, return_counts=True) 
n_mice = len(unique_names)
mice_data = dict(zip(unique_names, np.empty([n_mice,1])))

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
'''
Get: 
    'contrast_left'
    'contrast_right'
    'response'
''' 
dat_categ = ['contrast_left', 'contrast_right', 'response']
mice_data = {}
n_session = 0
n_mice = 0
for i, val in enumerate(name_counts): # loop for each mice
    print(unique_names[i], val)
    for i2 in range(val): # loop for number of sessions for each mice
        
        if n_session == 0 or n_session == n_mice:
            for categ in dat_categ:
                dat = alldat[n_session][categ]
                mice_data[(unique_names[i], categ)] = dat
            # dat = alldat[n_session]['response']
            # mice_data[(unique_names[i], 'response')] = dat
        else:
            for categ in dat_categ:
                dat = alldat[n_session][categ]
                prev_dat = mice_data[(unique_names[i], categ)]
                mice_data[(unique_names[i], categ)] = np.append(prev_dat,dat)
            
        print(n_session, len(alldat[n_session][categ]))
        
        n_session +=1
    n_mice += val
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
# ====================== iii) Analyze & Plot Stuff ======================
import function_analysis_by_mice as fun_spc
cont_diff = fun_spc.get_contDiff(mice_data, unique_names)

# ===== Check with Figures =====
# name = unique_names[3]
# fun_spc.plot_resp_contDiff(mice_data, name, cont_diff, saveplot=False)
# for name in unique_names:
#     fun_spc.plot_resp_contDiff(mice_data, name, cont_diff, saveplot=True)

right = {}
for name in unique_names:
    right[name] = fun_spc.get_rightward(mice_data, cont_diff, name)
# fun_spc.plot_rightward(right, cont_diff, unique_names, saveplot=False)

    idx_RL, right_levels = fun_spc.get_right_history(mice_data, cont_diff, name)


    fun_spc.plot_psychometric(mice_data, cont_diff, name, idx_RL, right_levels, right[name], savefig=True)

