# File containing a variety of utility functions for interacting with data
# Most of these functions are for dealing with data generated from using Azure scripts,
# so they have very specific functionality

import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score


def process_multimodel_data(filelist, data_path):  
    mean_acc = []
    min_acc = []
    max_acc = []
    var_acc = []
    stdev_acc = []

    mean_traj_count = []
    min_traj_count = []
    max_traj_count = []
    var_traj_count = []
    stdev_traj_count = []

    mean_frames = []
    mean_dist_tot = []
    mean_dist_net = []
    jacc_list = []

    for file in filelist:
        df = pd.read_csv(data_path + file)
        mean_acc.append(df['Accuracies'].mean())
        min_acc.append(df['Accuracies'].min())
        max_acc.append(df['Accuracies'].max())
        var_acc.append(df['Accuracies'].var())
        stdev_acc.append(df['Accuracies'].std())

        mean_traj_count.append(df['Trajectory Count'].mean())
        min_traj_count.append(df['Trajectory Count'].min())
        max_traj_count.append(df['Trajectory Count'].max())
        var_traj_count.append(df['Trajectory Count'].var())
        stdev_traj_count.append(df['Trajectory Count'].std())

        raw_frames = extract_string_from_df(df, 'Frames')
        mean_frames.append(np.array(raw_frames).mean())

        raw_dist_tot = process_dist_cols(df, 'dist_tot', nan_strat=None)
        mean_dist_tot.append(np.array(raw_dist_tot).mean())
        raw_dist_net = process_dist_cols(df, 'dist_net', nan_strat=None)
        mean_dist_net.append(np.array(raw_dist_net).mean())

        true_labels = process_labels(df, 'True Labels')
        preds = process_labels(df, 'Preds')
        jacc_score = jaccard_score(true_labels, preds, average=None)
        jacc_list.append(jacc_score)
           
        result = {'mean_acc': mean_acc, 
              'min_acc': min_acc, 
              'max_acc': max_acc, 
              'var_acc': var_acc, 
              'stdev_acc': stdev_acc, 
              'mean_traj_count': mean_traj_count, 
              'min_traj_count': min_traj_count,
              'max_traj_count': max_traj_count,
              'var_traj_count': var_traj_count,
              'stdev_traj_count': stdev_traj_count,
              'mean_frames': mean_frames,
              'mean_dist_tot': mean_dist_tot,
              'mean_dist_net': mean_dist_net,
              'jacc_scores': jacc_score
            #   'min_frames': min_frames,
            #   'max_frames': max_frames,
            #   'var_frames': var_frames,
            #   'stdev_frames': stdev_frames
              }
    return result


def extract_string_from_df(df, col_name):
    mean_frames = []



    subset_mean_frames_list = []

    for i in range(0,len(df)):
        frames_string = df[col_name][i][1:(len(df[col_name][i])-1)]
        frames_list = frames_string.split(",")
        list_of_floats = []
        #print(i)

        for val in frames_list:
            list_of_floats.append(float(val))
            #print(float(val))
        frames_array = np.array(list_of_floats)
        #print()
        mean_of_frames = frames_array.mean()
        
        subset_mean_frames_list.append(mean_of_frames)
            
            
            
        subset_mean_frames_array = np.array(subset_mean_frames_list)
        
        mean_frames.append(subset_mean_frames_array.mean())
    return mean_frames

    

def process_labels(df, col_name):
        list_of_ints = []
        for i in range(0,len(df)):
                frames_string = df[col_name][i][1:(len(df[col_name][i])-1)]
                frames_lines = frames_string.splitlines()
                for line in frames_lines:
                        if col_name == 'True Labels':
                                line = line.split('. ')
                        elif col_name == 'Preds':
                                line = line.split(', ')
                        for val in line:
                                if len(val) == 1:
                                        list_of_ints.append(int(val))
                                elif val[0] == ' ':
                                        val = val[1]
                                        list_of_ints.append(int(val))
                                else:
                                        val = val[0]
                                        list_of_ints.append(int(val))
        return list_of_ints


def process_dist_cols(df, col_name, nan_strat):
    dist_tot_vals = []
    for i in range(len(df)):
        thing = df[col_name][i][1:(len(df[col_name][i])-1)]
        thing = thing.splitlines()
        thing = thing[0].split(', ')
        dist_tot_sublist = []
        for val in thing:
            #print((val))
            if val != 'nan':
                dist_tot_sublist.append(float(val))
            elif nan_strat == 'zeros':
                    dist_tot_sublist.append(0.0)
        dist_tot_vals.append(np.array(dist_tot_sublist).mean())
    return dist_tot_vals