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

def plot_resp_contDiff(dat, cont_diff, n_session, saveplot=False):
	'''
	Inputs:
		* dat:
		* cont_diff:
	Outputs:
		* No output. Only make one plot. 
	'''
	fig = plt.figure(figsize=(12,5))

	plt.plot(cont_diff,'ro-', label='contrast difference')
	plt.plot(dat['response'],'bo', label='responses: right(+1) and left(-1)')
	plt.title('Session: %1.0f, '%n_session + dat['mouse_name'], fontsize=14)
	plt.legend()
	plt.show()

	if saveplot:
		print('Saving a figure...')
		plt.savefig('resp_contDiff_'+str(n_session)+'.png')
		plt.close(fig)

def plot_1psychometric(cont_diff, rightward, n_session, dat, saveplot=False):
	'''
    Description: Plot a psychometric curve for 'all' data.
    Inputs:
        'cont_diff': contrast difference (nine values).
        'rightward': % rightward.
        'n_session': session number.
        'dat': data for the selected session.
	'''
	xdata = np.unique(cont_diff)
	ydata = rightward

	plt.figure(figsize=(6,4))

	plt.plot(xdata, ydata,'o-', label=n_session)
	plt.xlabel('Contrast difference')
	plt.ylabel('Rightward (%)')
	plt.title('Session: %1.0f, '%n_session + dat['mouse_name'], fontsize=16)
	plt.legend(loc='upper left', fontsize=10)
	plt.grid()
	plt.show()

	if saveplot:
		print('Saving a figure...')
		plt.savefig('psychometric1_'+str(n_session)+'.png')
		plt.close()
    
def plot_psychometric(cont_diff, rightward, response, right_levels, idx_RL, n_session, dat, srcdir, savefig=False):
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
	xdata = np.unique(cont_diff)
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

	plt.xlabel('Contrast difference')
	plt.ylabel('Rightward (%)')
	plt.title('Session: %1.0f, '%n_session + dat['mouse_name'], fontsize=16)
	plt.legend(loc='upper right', fontsize=10)
	plt.grid()
	fig.show()

	if savefig:
		print('Saving a figure to', srcdir)
		fig.savefig('resp_dir_'+str(n_session)+'.png')
		plt.close(fig)

def plot_bars(langs, diff_mean, diff_std, srcdir, n_session, saveplot=False):
	'''
	This function makes a bar plot with plt.bar().

	Inputs:
		* langs:
		* diff_mean:
		* diff_std:
		* srcdir:
		* n_session:
	'''

	# make a bar plot for one session
	fig = plt.plot(figsize=(8,6))
	# ax = fig.add_axes([0.1,0.1,.8,.8]) # ([bottom left, top right])
	# ax.bar(langs, values, yerr = std_vals, capsize=10)
	plt.bar(langs, diff_mean, yerr=diff_std, capsize=5)
	plt.title("Session: %1.0f" %n_session)
	plt.xlabel("Previous difficulty & choice")
	# plt.ylabel("Probability of Rightward Choice (%)")
	plt.ylabel("Mean difference in %right w.r.t ALL")
	# ax.set_ylim([0,100])
	plt.grid()
	plt.show

	if saveplot:
		print('Saving a figure to', srcdir)
		plt.savefig('bar_'+str(n_session)+'.png')
		plt.close(fig)

# def plot_psych_correct_response(saveplot=False):
    
def plot_stim_dir(n_session, dat, cont_diff, right_levels, keys, n_trials, saveplot=False):
    '''

    Parameters
    ----------
    n_session : TYPE
    dat : TYPE
    cont_diff : TYPE
    rightward : TYPE
    r_diffr : TYPE
    r_easyr : TYPE
    r_easyl : TYPE
    r_diffl : TYPE

    Returns
    -------
    None. Only a figure.

    '''
    fig = plt.figure(figsize=(7,5))

    xdata = np.unique(cont_diff)
    plt.plot(xdata, right_levels['all'],'ro-', 
             label='all (%1.0f)'%n_trials['all'], alpha=0.5, linewidth=3)
    plt.plot(xdata, right_levels['hard_cr'],'D-', color=colors['darkviolet'],
             label='hard_r (%1.0f)'%n_trials['hard_cr'], alpha=0.5, linewidth=2)
    plt.plot(xdata, right_levels['easy_cr'],'^:', color=colors['violet'],
             label='easy_r (%1.0f)'%n_trials['easy_cr'], alpha=0.5, linewidth=2)
    plt.plot(xdata, right_levels['zero'],'x--', color=colors['deeppink'],
             label='zero (%1.0f)'%n_trials['zero'], linewidth=2)
    plt.plot(xdata, right_levels['easy_cl'],'^:', color=colors['skyblue'],
             label='easy_l (%1.0f)'%n_trials['easy_cl'], alpha=0.5, linewidth=2)
    plt.plot(xdata, right_levels['hard_cl'],'D-', color=colors['dodgerblue'], 
             label='diff_l (%1.0f)'%n_trials['hard_cl'], alpha=0.5, linewidth=2)

    plt.xlabel('Contrast difference')
    plt.ylabel('Rightward (%)')
    plt.title('Session: %1.0f, '%n_session + dat['mouse_name'])
    plt.legend(fontsize=8)
    plt.grid()
    plt.show()

    if saveplot:
        print('Saving a figure...')
        plt.savefig('stim_dir_'+str(n_session)+'.png')
        plt.close(fig)



