# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:28:12 2020

@author: Akihiro Yamaguchi

This script contains a list of functions to generate figures.
They are used in 'main_behavior_analysis.py' and 'batch_behavior_analysis.py'.

"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-    
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-* Analysis Functions *-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
def get_contDiff(mice_data, unique_names):
	'''
	Inputs
		'mice_data': dictionary of mice names, contrast_left, contrast_right, response.
		'unique_names': array of strings containing mice names.
	Returns
		'cont_diff': contrast difference between left and right.
	'''
	cont_diff = {}

	for mice in unique_names:
		cont_diff[mice] = mice_data[mice, 'contrast_left']-mice_data[mice, 'contrast_right']

	return cont_diff

def get_rightward(mice_data, cont_diff, name):
	'''
	Inputs:
		'cont_diff': contrast difference between left and right visual stimuli. (dictionary)
		'name': name of the mice.

	'''
	rightward = np.zeros(len(np.unique(cont_diff[name]))) * np.nan
	unique, counts = np.unique(cont_diff[name], return_counts=True) # check the contrast differences and the number of occurences

	for i, val in enumerate(np.unique(cont_diff[name])):
		resp = mice_data[name, 'response'][cont_diff[name]==val]
		rightward[i] = np.count_nonzero(resp<0) / counts[i]*100 # '-1' for 'right' choice

	return rightward

def get_right_history(mice_data, cont_diff_dict, name):
    '''
    This function make a dictionary of the rightward choice for previous trial 
    difficulty and response (left/right). 

    Inputs:
    	* mice_data: mice specific dataset labeled by the names.
        * cont_diff: mice specific contrast difference.
        * name: mice name.

    Outputs:
        * idx_RL: indices of [...]
        * right_levels: %rightward for each task difficulty ('easy', 'hard', or 'zero')
                        and response ('r' or 'l').

    Warning: the indices of the keys in 'right_levels' and 'idx_RL' matches!
    So be careful when you change their order.
    '''

    cont_diff = cont_diff_dict[name]

    # Keys of the dictionary. Corresponding idx = 0 to 6.
    keys = ['hard_l', 'easy_l', 'zero_l', 'all', 'zero_r', 'easy_r', 'hard_r'] 
    n = len(keys)
    vals = np.empty([len(keys),9]) # len(keys) should be 7, there are 9 contrast levels.
    vals[:] = np.nan

    # Construct a dictionary
    right_levels = dict(zip(keys,vals))

    response=mice_data[name, 'response'] # all responses (340 for session 11)

    # Indices of right/left choice
    idx_choice_r = np.array([i for i, x in enumerate(response<0) if x]) # trial number of right choice
    idx_choice_l = np.array([i for i, x in enumerate(response>=0) if x]) # trial number of left choice (includes 'no go')

    # List of empty integer arrays. Prepare to store lists of indices. (n=7)
    idx_RL = [[np.empty(0, int)]*1]*n

    # Assign indices of the LEFT choices for each difficulty
    for i, idx in enumerate(idx_choice_l):
        if idx == (len(response)-1): continue # Discard the last trial
        
        if ((abs(cont_diff[idx]) == 0.25) | (abs(cont_diff[idx]) == 0.5)): # hard trials
            idx_RL[0] = np.append(idx_RL[0], idx)
        elif ((abs(cont_diff[idx]) == 1) | (abs(cont_diff[idx]) == 0.75)): # easy trials
            idx_RL[1] = np.append(idx_RL[1], idx)
        if (abs(cont_diff[idx]) == 0): # zero trials
            idx_RL[2] = np.append(idx_RL[2], idx)

    # For "ALL" trials (340 trials for session 11)
    idx_RL[3] = np.linspace(0,len(response)-1,len(response)).astype(int) # Just an array of all the indices

    # Assign indices of the RIGHT choices for each difficulty
    for i, idx in enumerate(idx_choice_r):
        if idx == (len(response)-1): continue # Discard the last trial
        
        if (abs(cont_diff[idx]) == 0): # zero trials
            idx_RL[4] = np.append(idx_RL[4], idx)
        elif ((abs(cont_diff[idx]) == 1) | (abs(cont_diff[idx]) == 0.75)): # easy trials
            idx_RL[5] = np.append(idx_RL[5], idx)
        elif ((abs(cont_diff[idx]) == 0.25) | (abs(cont_diff[idx]) == 0.5)): # hard trials
            idx_RL[6] = np.append(idx_RL[6], idx)

    # Fill the %right choice for each of 9 contrast differences
    for j in range(n-1):
        # Skip this process for 'all' case (index of 3)
        if j < 3:
            # check the contrast differences (unique) and the number of occurences (counts)
            unique, counts = np.unique(cont_diff[idx_RL[j]+1], return_counts=True)

            print('unique cont_diff size:', np.unique(cont_diff[idx_RL[j]+1]).size, keys[j])
            unq_values = np.unique(cont_diff[idx_RL[j]+1]) # unique values for the 'current' trials.
            for i, val in enumerate(unq_values): # Assumption: cont_diff[...] has all the 9 contrast differences.
            #     print(i,':', val)
                resp = mice_data[name, 'response'][idx_RL[j]+1][cont_diff[idx_RL[j]+1]==val]
                array_indx = int(4*val + 4) # convert the contrast difference value to the corresponding array index
                right_levels[keys[j]][array_indx] = np.count_nonzero(resp<0) / counts[i]*100 # right choice: i = -1
        
        elif j >=3:
            j += 1
            # check the contrast differences and the number of occurences
            unique, counts = np.unique(cont_diff[idx_RL[j]+1], return_counts=True)

            print('unique cont_diff size:', np.unique(cont_diff[idx_RL[j]+1]).size, keys[j])
            for i, val in enumerate(np.unique(cont_diff[idx_RL[j]+1])): # Assumption: cont_diff[...] has all the 9 contrast differences.
            #     print(i,':', val)
                resp = mice_data[name,'response'][idx_RL[j]+1][cont_diff[idx_RL[j]+1]==val]
                array_indx = (4*val + 4).astype(int) # convert the contrast difference value to the corresponding array index
                right_levels[keys[j]][array_indx] = np.count_nonzero(resp<0) / counts[i]*100 # right choice: i = -1

    right_levels["all"] = get_rightward(mice_data, cont_diff_dict, name)
    print('all trial data is added.')

    return idx_RL, right_levels

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**    
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-* Plot Functions *-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**    

def plot_rightward(rightward, cont_diff, names, saveplot=False):
	'''
	INPUTS:
		'rightward': dictionary of %rightward response for each mice.
		'cont_diff': contrast difference between left and right stimuli, categorized by dict.
		'names': list of names.
		'saveplot': Optional. set 'True' to save the figure.
	'''

	plt.figure(figsize=(12,8))

	for name in names:
		xdata = np.unique(cont_diff[name])
		ydata = rightward[name]
		plt.plot(xdata, ydata,'o-', label=name+' (%1.0f)'%len(cont_diff[name]))

	plt.xlabel('Contrast difference (contrast_left - contrast_right)', fontsize=12)
	plt.ylabel('Rightward (%)', fontsize=12)
	# plt.title(name, fontsize=16)
	plt.legend(loc='upper right', fontsize=10)
	plt.grid()
	plt.show()

	if saveplot:
		print('Saving a figure...')
		plt.savefig(name+'(%1.0f)'%(len(cont_diff[name]))+'_rightward'+'.png')
		plt.close(fig)

def plot_psychometric(mice_data, cont_diff, name, idx_RL, right_levels, rightward, savefig=False):
	'''
	Make a psychometric function.
	Inputs:
		'cont_diff':
		'rightward': % rightward 
		'response': response ('-1' for right, '0' for no go, '1' for left) 
		'right_levels': 
		'idx_RL': indices of response+difficulty. (idx_RL[0] for hard_l, [6] for hard_r)
		'n_session': session number.
		'dat': data for a selected session.

	'''
	xdata = np.unique(cont_diff[name])
	ydata = rightward

	fig = plt.figure(figsize=(8,6))

	plt.plot(xdata, ydata,'ro-', label='All (%1.0f)'%len(idx_RL[3]), alpha=0.5, linewidth=3)

	plt.plot(xdata, right_levels['hard_r'],'D-',  color=colors['darkviolet'], 
	         label='hard_r (%1.0f)'%(idx_RL[6].size), linewidth=2)
	plt.plot(xdata, right_levels['easy_r'],'^:',  color=colors['violet'], 
	         label='easy_r (%1.0f)'%(idx_RL[5].size), linewidth=2)
	plt.plot(xdata, right_levels['zero_r'],'x--', color=colors['lime'], 
	         label='zero_r (%1.0f)'%(idx_RL[4].size), linewidth=2, alpha=0.5)
	plt.plot(xdata, right_levels['zero_l'],'x--', color=colors['deeppink'], 
	         label='zero_l (%1.0f)'%(idx_RL[2].size), linewidth=2, alpha=0.5)
	plt.plot(xdata, right_levels['easy_l'],'^:',  color=colors['skyblue'], 
	         label='easy_l (%1.0f)'%(idx_RL[1].size), linewidth=2)
	plt.plot(xdata, right_levels['hard_l'],'D-',  color=colors['dodgerblue'], 
	         label='hard_l (%1.0f)'%(idx_RL[0].size), linewidth=2)

	plt.xlabel('Contrast difference (contrast_left - contrast_right)', fontsize=12)
	plt.ylabel('Rightward (%)', fontsize=12)
	plt.title(name + ' (%1.0f trials)'%len(cont_diff[name]), fontsize=16)
	plt.legend(loc='upper right', fontsize=10)
	plt.grid()
	fig.show()

	if savefig:
		print('Saving a figure...')
		fig.savefig(name+'_(%1.0f_trials)'%len(cont_diff[name])+'.png')
		plt.close(fig)

def plot_resp_contDiff(mice_data, name, cont_diff, saveplot=False):
	'''
	Inputs:
		* dat:
		* cont_diff:
	Outputs:
		* No output. Only make one plot. 
	'''

	width = int(0.035*len(cont_diff[name]))
	fig = plt.figure(figsize=(width,5))

	plt.plot(cont_diff[name],'ro-', label='contrast difference')
	plt.plot(mice_data[name, 'response'],'bo', label='responses: right(-1) and left(+1)')
	plt.title('Mice: '+ name + ' (%1.0f trials)'%(len(cont_diff[name])), fontsize=14)
	plt.ylabel('Contrast difference & Response')
	plt.xlabel('Trial Number')
	plt.legend()
	plt.grid()
	plt.show()

	if saveplot:
		print('Saving a figure...')
		plt.savefig(name+'(%1.0f)'%(len(cont_diff[name]))+'_cont_diff'+'.png')
		plt.close(fig)

