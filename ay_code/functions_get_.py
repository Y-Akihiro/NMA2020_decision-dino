# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 11:11:19 2020

@author: Akihiro Yamaguchi

List of functions for 'main_behavior_analysis.py'

"""
import os, requests
import numpy as np

from IPython.display import HTML
import random

def download_data():
    '''
    A function to downlaod data files from osf website. Only need to run once.
    Returns 'alldat'
    '''
    fname = []
    for j in range(3):
        fname.append('steinmetz_part%d.npz'%j)
    url = ["https://osf.io/agvxh/download"]
    url.append("https://osf.io/uv3mw/download")
    url.append("https://osf.io/ehmw2/download")
    
    for j in range(len(url)):
        if not os.path.isfile(fname[j]):
            try:
                r = requests.get(url[j])
            except requests.ConnectionError:
                print("!!! Failed to download data !!!")
            else:        
                if r.status_code != requests.codes.ok:
                    print("!!! Failed to download data !!!")
                else:
                    with open(fname[j], "wb") as fid:
                        fid.write(r.content)
    # Data loading
    alldat = np.array([])
    for j in range(len(fname)):
        alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))

    return alldat

def get_colors():
    '''
    Returns
    -------
    colors : 156 different colors defined by names (dict)
    by_hsv : 156 colors defined by RGB values (list) 
    sorted_names : sorted names of 156 colors (str)

    '''
    # ===================== for plotting =====================
    # Sort colors by hue, saturation, value and name.
    from matplotlib import colors as mcolors
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    
    return colors, by_hsv, sorted_names

def hide_toggle(for_next=False):
    '''
    This is a function to reduce space for jupyter notebook. 
    '''
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )

    return HTML(html)

def load_names(fname=[]):
    del fname
    fname = []
    for j in range(3):
        fname.append('steinmetz_part%d.npz'%j)
    url = ["https://osf.io/agvxh/download"]
    url.append("https://osf.io/uv3mw/download")
    url.append("https://osf.io/ehmw2/download")
    
    for j in range(len(url)):
        if not os.path.isfile(fname[j]):
            try:
                r = requests.get(url[j])
            except requests.ConnectionError:
                print("!!! Failed to download data !!!")
            else:        
                if r.status_code != requests.codes.ok:
                    print("!!! Failed to download data !!!")
                else:
                    with open(fname[j], "wb") as fid:
                        fid.write(r.content)
    return fname

def load_data(n_session, alldat):
    '''
    Inputs:
        * n_session: session number
        * alldat: 
    '''
    dat = alldat[n_session]

    # groupings of brain regions
    regions = ["vis ctx", "thal", "hipp", "other ctx", "midbrain", "basal ganglia", "cortical subplate", "other"]
    brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                    ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                    ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                    ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"], # non-visual cortex
                    ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                    ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia 
                    ["BLA", "BMA", "EP", "EPd", "MEA"] # cortical subplate
                    ]

    nareas = 4 # only the top 4 regions are in this particular mouse
    NN = len(dat['brain_area']) # number of neurons
    barea = nareas * np.ones(NN, ) # last one is "other"
    for j in range(nareas):
        barea[np.isin(dat['brain_area'], brain_groups[j])] = j # assign a number to each region
        
    return dat, barea, NN, regions, brain_groups, nareas

def get_rightward(dat, cont_diff):
    """
    Inputs: 
        * dat - data 
        * cont_diff - contrast difference between left and right
    Return:
        * rightward - % of mice respond right for each contrast difference
        * response of -1 corresponds to 'right' and 1 to 'left' choices
    """
    rightward = np.zeros(len(np.unique(cont_diff)))
    unique, counts = np.unique(cont_diff, return_counts=True) # check the contrast differences and the number of occurences

    for i, val in enumerate(np.unique(cont_diff)):
        resp = dat['response'][cont_diff==val]
        rightward[i] = np.count_nonzero(resp<0) / counts[i]*100 # '-1' for 'right' choice
    
    # print(rightward)
    return rightward

def logistic_function(x,x0, y0, a,b):
    '''
    Four-parameter logistic function for model fitting
    '''
    return y0 + a/(1+np.exp(-(x-x0)/b))

def get_right_hist_1(dat, cont_diff):
    """
    This function computes %rightward choice with previous difficulty and stimulus direction.
    
    Inputs: 
        * dat - trial data for a selected session.
        * cont_diff - contrast difference between left and right.
    Return:
        'right_levels': % rightward for each difficulty and previous correct direction.
        'keys': keys used in the dictionary.
        'n_trials': number of trials in each difficulty/direction.
    """
    
    response = dat['response']
    keys = ['hard_cl', 'easy_cl', 'easy_cr', 'hard_cr', 'zero', 'all'] # corresponding idx = 0 to 5.
    n = len(keys)
    vals = np.empty([n,9]) # len(keys) should be 5.
    # Prepare an empty dictionary to store % right for each stim_direction+difficulty
    vals[:] = np.nan

    bool_vals = np.empty([n, len(cont_diff)])
    bool_vals[:] = np.nan
    # Construct a dictionary
    right_levels = dict(zip(keys,vals))
    # idx_bools = dict(zip(keys, bool_vals))
    
    idx = dict(zip(keys, np.empty([n,1])))
    resp = dict(zip(keys,np.empty([n,1])))
    contrast = dict(zip(keys,np.empty([n,1])))
    # unq = dict(zip(keys,np.empty([n,1])))
    # ocr = dict(zip(keys,np.empty([n,1])))
    
    n_trials = dict(zip(keys,np.empty([n,1])))
    # defn = np.array([[-1], [-0.25], [0.25], [1]])
    # # defn = np.array([[-1, -0.75], [-0.5,-0.25], [0.25, 0.5], [0.75, 1]]) # diffeernt definition
    # diffside_defn = dict(zip(keys,defn))
    # =====================================================================
    
    # Get the contrast differences and the number of occurences (trials)
    unique, counts = np.unique(cont_diff, return_counts=True) 
    print('unique:', unique)
    print('counts:', counts)
    
    # Get the indices of previous stimulus direction + difficulty
    idx['easy_cl'] = np.where(cont_diff==unique[0])[0]
    idx['hard_cl'] = np.where(cont_diff==unique[3])[0]
    idx['hard_cr'] = np.where(cont_diff==unique[5])[0]
    idx['easy_cr'] = np.where(cont_diff==unique[8])[0]
    idx['zero'] = np.where(cont_diff == 0)[0]
    idx['all'] = np.linspace(0,len(response)-1,len(response)).astype(int) 
    
    for key in keys:
        n_trials[key] = len(idx[key])
        
        # Get the 'current trial' response for each difficulty + stimulus direction history.
        # Also get current contrast difference (difficulty/direction)
        if key=='all':
            resp[key] = response
            contrast['all'] = cont_diff
        else:
            # Check the index
            if len(np.where(idx[key]==len(response)-1)[0])==1:
                idx[key] = np.delete(idx[key],-1)
                print('The last index in', key, 'is deleted.')
            resp[key] = response[idx[key] + 1]
            contrast[key] = cont_diff[idx[key]+1]
    
    # Compute % rightward for each key (difficulty_stimulus direction)
    for key in keys:
        print(key)
        if key=='all':
            right_levels[key] = get_rightward(dat, cont_diff)
        else:
            # Compute % rightward for each of 9 contrasts
            i_temp = []
            for i, val in enumerate(unique):
                arr = (idx[key]+1)[np.where(cont_diff[idx[key]+1]==val)[0]]
                if len(arr) == 0:
                    print('arr is empty! check val=',val)
                    i_temp.append(i)
                right_levels[key][i] = np.divide(sum(response[arr]<0),len(arr), out=np.zeros(1), where=len(arr)!=0)*100
            right_levels[key][i_temp]=np.nan # replace the empty element with NaN
            
    # unq['easy_cl'], ocr['easy_cl'] = np.unique(contrast['easy_cl'], return_counts=True)

    return right_levels, keys, n_trials

def get_right_hist_2(dat, cont_diff):
    """
    Inputs: 
        * dat - trial data 
        * cont_diff - contrast difference between left and right
    Return:
        * reasy_l - % of mice respond right for each contrast difference, previous trial is easy left
        * rdiff_l - % of mice respond right for each contrast difference, previous trial is difficult left
        * reasy_r - % of mice respond right for each contrast difference, previous trial is easy right
        * rdiff_r - % of mice respond right for each contrast difference, previous trial is difficult right
    """
    unique, counts = np.unique(cont_diff, return_counts=True) # check the contrast differences and the number of occurences

    n_el = n_dl = n_er = n_dr = n_zero = 0

    # Define easy/difficult left/right (boolean array)
    easy_l = (cont_diff==-1) + (cont_diff==-0.75)
    diff_l = (cont_diff==-0.25) + (cont_diff==-0.5)
    diff_r = (cont_diff==0.25) + (cont_diff==0.5)
    easy_r = (cont_diff==1) + (cont_diff==0.75)

    reasy_l = np.zeros((easy_l.sum(),9,2)) # 59 x 9
    rdiff_l = np.zeros((diff_l.sum(),9,2)) # 52 x 9
    reasy_r = np.zeros((easy_r.sum(),9,2)) # 41 x 9
    rdiff_r = np.zeros((diff_r.sum(),9,2)) # 62 x 9
    rzero = np.zeros(((cont_diff==0).sum(),9,2)) # 126 x 9

    # Check the number of trials for each difficulties 
    n_trials = np.zeros(6)
    n_trials[0] = rzero.shape[0] 
    n_trials[1] = reasy_l.shape[0]
    n_trials[2] = rdiff_l.shape[0]
    n_trials[3] = rdiff_r.shape[0]
    n_trials[4] = reasy_r.shape[0]
    n_trials[5] = dat['spks'].shape[1]
    
    for i in range(len(dat['response'])-1):
        hist = cont_diff[i]                                 # previous trial difficulty (size) & direction (sign)
        idx_cont = np.where(unique==cont_diff[i+1])[0][0]   # current trial difficulty and direction label (9 unique labels)

        if hist in unique[0:2]: # easy left
            reasy_l[n_el, idx_cont,0] = dat['response'][i+1]
            reasy_l[0,idx_cont,1] +=1
            n_el += 1
        elif hist in unique[2:4]: # difficult left
            rdiff_l[n_dl, idx_cont,0] = dat['response'][i+1]
            rdiff_l[0, idx_cont,1] += 1
            n_dl += 1
        elif hist in unique[5:7]: # difficult right
            rdiff_r[n_dr, idx_cont,0] = dat['response'][i+1]
            rdiff_r[0, idx_cont,1] += 1
            n_dr += 1
        elif hist in unique[7:9]: # easy right
            reasy_r[n_er, idx_cont,0] = dat['response'][i+1]
            reasy_r[0, idx_cont,1] += 1
            n_er += 1
        elif hist == 0:
            rzero[n_zero, idx_cont,0] = dat['response'][i+1]
            rzero[0, idx_cont,1] += 1
            n_zero += 1
        else:
            hist += 1 # some action. (whatever is fine to pass)
    #         print('Check: something is wrong')

    # Use np.divide(a, b, out=np.zeros(a.shape), where=b!=0) to avoid 0 division error
    r_easyr = np.divide(np.count_nonzero(reasy_r[:,:,0]==1,axis=0),
                        reasy_r[0,:,1], 
                        out=np.zeros(np.count_nonzero(reasy_r[:,:,0]==1,axis=0).shape), 
                        where=(reasy_r[0,:,1]!=0)) * 100
    r_easyl = np.divide(np.count_nonzero(reasy_l[:,:,0]==1,axis=0),
                        reasy_l[0,:,1], 
                        out=np.zeros(np.count_nonzero(reasy_l[:,:,0]==1,axis=0).shape), 
                        where=(reasy_l[0,:,1]!=0)) * 100
    r_diffr = np.divide(np.count_nonzero(rdiff_r[:,:,0]==1,axis=0),
                        rdiff_r[0,:,1], 
                        out=np.zeros(np.count_nonzero(rdiff_r[:,:,0]==1,axis=0).shape), 
                        where=(rdiff_r[0,:,1]!=0)) * 100
    r_diffl = np.divide(np.count_nonzero(rdiff_l[:,:,0]==1,axis=0),
                        rdiff_l[0,:,1], 
                        out=np.zeros(np.count_nonzero(rdiff_l[:,:,0]==1,axis=0).shape), 
                        where=(rdiff_l[0,:,1]!=0)) * 100
    r_zero = np.divide(np.count_nonzero(rzero[:,:,0]==1,axis=0),
                       rzero[0,:,1], 
                       out=np.zeros(np.count_nonzero(rzero[:,:,0]==1,axis=0).shape), 
                       where=(rzero[0,:,1]!=0)) * 100
        
    return r_easyr, r_easyl, r_diffr, r_diffl, r_zero, n_trials

def get_task_difference(n_session, dat):
    '''
    Description: TBA

    Inputs:
        * n_session: session number
        * dat: session specific dataset\n
    Outputs:
        * dt:
        * NT:
        * cont_diff: contrast difference
        * abs_task_diff: 
        * dtask_diff: 
        * dtdiff:
    '''

    dt = dat['bin_size']                  # binning at 10 ms
    NT = dat['spks'].shape[-1]

    l_cont = dat['contrast_left']         # contrast left
    r_cont = dat['contrast_right']        # contrast right

    cont_diff = l_cont - r_cont              # contrast difference: right if negative, left if positive
    abs_task_diff = np.abs(l_cont - r_cont)  # absolute contrast difference
    dtask_diff = np.diff(abs_task_diff)      # change in contrast difference (current - previous)
    dtdiff = np.insert(dtask_diff, 0, 0)     # adjust the array size

    return dt, NT, cont_diff, abs_task_diff, dtask_diff, dtdiff

def get_right_history(dat, cont_diff):
    '''
    This function make a dictionary of the rightward choice for previous trial 
    difficulty and response (left/right). 

    Inputs:
    	* dat: session specific dataset obtained from 'load_data()'
        * cont_diff: session specific contrast difference from 'get_task_difference()'

    Outputs:
        * idx_RL: indices of [...]
        * right_levels: %rightward for each task difficulty ('easy', 'hard', or 'zero')
                        and response ('r' or 'l').

    Warning: the indices of the keys in 'right_levels' and 'idx_RL' matches!
    So be careful when you change their order.
    '''

    # keys = ['easy_r', 'hard_r', 'zero_r', 'easy_l', 'hard_l', 'zero_l', 'all']
    # the order of the keys I like to use:
    keys = ['hard_l', 'easy_l', 'zero_l', 'all', 'zero_r', 'easy_r', 'hard_r'] # corresponding idx = 0 to 6.
    n = len(keys)
    vals = np.empty([len(keys),9]) # len(keys) should be 7.
    vals[:] = np.nan

    # Construct a dictionary
    right_levels = dict(zip(keys,vals))

    response=dat['response'] # all responses (340 for session 11)

    # Indices of right/left choice
    idx_choice_r = np.array([i for i, x in enumerate(response<0) if x]) # trial number of right choice
    idx_choice_l = np.array([i for i, x in enumerate(response>=0) if x]) # trial number of left choice (includes 'no go')

    # Task difficulty of the right/left choice (This is already taken care of by the 'enumerate' part below.)
    # level_r = abs(cont_diff[idx_choice_r]) # contrast difficulty (abs of contrast difference) of right choice trials 
    # level_l = abs(cont_diff[idx_choice_l]) # contrast difficulty (abs of contrast difference) of left choice trials 

    idx_RL = [[np.empty(0, int)]*1]*n # List of empty integer arrays. Prepare to store lists of indices. (n=7)

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
                resp = dat['response'][idx_RL[j]+1][cont_diff[idx_RL[j]+1]==val]
                array_indx = int(4*val + 4) # convert the contrast difference value to the corresponding array index
                right_levels[keys[j]][array_indx] = np.count_nonzero(resp<0) / counts[i]*100 # right choice: i = -1
        
        elif j >=3:
            j += 1
            # check the contrast differences and the number of occurences
            unique, counts = np.unique(cont_diff[idx_RL[j]+1], return_counts=True)

            print('unique cont_diff size:', np.unique(cont_diff[idx_RL[j]+1]).size, keys[j])
            for i, val in enumerate(np.unique(cont_diff[idx_RL[j]+1])): # Assumption: cont_diff[...] has all the 9 contrast differences.
            #     print(i,':', val)
                resp = dat['response'][idx_RL[j]+1][cont_diff[idx_RL[j]+1]==val]
                array_indx = (4*val + 4).astype(int) # convert the contrast difference value to the corresponding array index
                right_levels[keys[j]][array_indx] = np.count_nonzero(resp<0) / counts[i]*100 # right choice: i = -1

    right_levels["all"] = get_rightward(dat, cont_diff)
    print('all trial data is added.')

    return idx_RL, right_levels

def get_bars_data(right_levels):
    '''
    This function creates a dictionary of previous trial response 
    & difficulty for 9 contrast differences.

    Inputs:
        * right_levels: 
    Outputs: 
        * langs: 7 difficulty & response keys.
        * diff_mean: mean of the differences for 9 contrast differences.
        * diff_std: standard deviation of the differences for 9 contrast differences.
    '''
    langs = list(right_levels.keys())
    values = np.zeros(len(langs))
    std_vals = np.zeros(len(langs))
    diff_mean = np.zeros(len(langs))
    diff_std = np.zeros(len(langs))

    for i, key in enumerate(langs):
        print('i=%1.0f'%i, key)

        values[i] = right_levels[key][~np.isnan(right_levels[key])].mean()
        std_vals[i] = right_levels[key][~np.isnan(right_levels[key])].std()
        diff_mean[i] = np.mean(right_levels['all'][~np.isnan(right_levels[key])] - right_levels[key][~np.isnan(right_levels[key])])
        diff_std[i] = np.std(right_levels['all'][~np.isnan(right_levels[key])] - right_levels[key][~np.isnan(right_levels[key])])
        langs[i] = key+'(%1.0f)'%len(right_levels[key][~np.isnan(right_levels[key])])
        # print('mean: ', str(diff_mean[i]))
        
    return langs, diff_mean, diff_std

def get_std_errors(dat, idx_RL, langs):
    '''
    Inputs:
        'dat'
        'idx_RL'
        'langs':
    Outputs:
        'std_error'
        'p1'
    '''
    response = dat['response']
    p = np.zeros(len(langs))
    std_error = np.zeros(len(langs))
    
    # Get n2 for 'all' data
    n2_r = sum(response==-1) # '-1' for rightward choice
    n2 = len(response)
    p1 = np.zeros(len(langs)) # %rightward for each direction+difficulty.
    p2 = n2_r/n2 # %rightward for 'all' trials

    for i, key in enumerate(langs):
        # Get n1 for a selected response/difficulty
        # idx_RL[#] #--> 0 to 6: hard_l, easy_l, zero_l, all, zero_r, easy_r, hard_r
        
        if i !=3:
            n1_r = sum(response[idx_RL[i]+1]==-1)
            n1 = len(response[idx_RL[i]+1]==-1)
        else:
            n1_r = sum(response[idx_RL[i]]==-1)
            n1 = len(response[idx_RL[i]]==-1)
            
        p1[i] = n1_r/n1 # %rightward for a task
        
        p[i] = (n1*p1[i] + n2*p2) / (n1 + n2)
        
        std_error[i] = np.sqrt(p[i]*(1-p[i]) * (1/n1 + 1/n2))
    
    return std_error, p1

def get_correctness(dat, cont_diff):
    '''
    Parameters
    ----------
    dat : TYPE
    cont_diff : TYPE

    Returns
    -------
    right_levels : TYPE
    n_trials : TYPE
    '''
    
    
    return right_levels, n_trials


