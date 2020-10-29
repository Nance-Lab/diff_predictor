import os
import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import csv
import boto3
import pandas as pd
import diff_classifier.aws as aws
from itertools import cycle


if 'core' not in sys.modules:
    import core


def load_data(folder, filenames=[], **kwargs):
    """
    Load data either through the system or through aws S3.

    Parameters
    ----------
    folder : string :
        desired folder to import files from
    filenames : list of strings :
        desired files to import

    Optional Parameters
    -------------------
    download_list_file : string :
        if using a textfile containing multiple filenames, use this to designate location
        of this file within the folder.
        ex: folder/download_file_names.txt
    tag : list of strings :
        if tagging a dataframe file with a variable, use this to tag each file. Will cycle
        through list if list reaches end and there are stile files in the filenames list
    bucket_name : string :
        if using aws S3, declare this variable as an S3 bucket to look through. This will
        trigger the function so that folder is the folder in the bucket and filenames are
        the filenames to download in the bucket

    """
    data = pd.DataFrame()
    tag = None
    if 'download_list_file' in kwargs:
        list_path = os.path.join(folder, kwargs['download_list_file'][0])
        assert os.path.isfile(list_path) and os.access(list_path, os.R_OK), \
            f'{list_path} does not exhist or can not be read'
        try:
            with open(list_path, 'r') as f:
                filenames = f.read().splitlines()
        except IOError as err:
            print(f"Could not read {f}: {err}")
    if 'tag' in kwargs:
        tag = cycle(kwargs['tag'])
    if 'bucket_name' in kwargs:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(kwargs['bucket_name'])
        for filename in filenames:
            if tag:
                file_tag = next(tag)
            else:
                file_tag = None
            try:
                file_path = os.path.join(folder, filename)
                print(file_path)
                aws.download_s3(file_path, filename, bucket_name=bucket)
                file_data = pd.read_csv(filename, encoding="ISO-8859-1", index_col='Unnamed: 0')
                if file_tag:
                    size = file_data.shape[0]
                    file_data['Tag'] = pd.Series(size*[file_tag], index=file_data.index)
                data = pd.concat([data, file_data])
                del file_data
            except IOError as err:
                print(f'Skipped!: {filename}: {err}')
        return data
    for filename in filenames:
        if tag:
            file_tag = next(tag)
        else:
            file_tag = None
        try:
            file_path = os.path.join(folder, filename)
            print(file_path)

            if file_tag:
                size = file_data.shape[0]

            data = pd.concat([data, file_data])
        except IOError as err:
            print(f'Skipped!: {filename}: {err}')
    return data


def get_files(path, keywords = ["features_ OR msd_"]):
    """
    Takes in a path and list of keywords. Returns a list of filenames
    that are within the path that contain one of the keyword in the list.
    Set keyword to "" to get all files in the path.
    
    Parameters
    ----------
    path : string
        file path
    keywords : string or [string] : ["features_ OR msd_"]
        keywords to look for in the file path. 
        
    Returns
    -------
    file_list : list
        list of files in the path
    """
    keywords = [i.split('OR') for i in list(keywords)]
    keywords = [list(map(lambda x:x.strip(), i)) for i in keywords]
    files = [f for f in listdir(path) if isfile(join(path, f))]
    file_list = []
    for filename in files:
        kwds_in = all(any(k in filename for k in ([keyword]*isinstance(keyword, str) or keyword)) for keyword in keywords)
        if (kwds_in):
            file_list.append(filename)
    return file_list


# Pre: Both files must exhist; Feature must be in the feature file
# Throws a FileNotFoundError exception if preconditions not met
#
# Adds a feature from produced features file to the track file.
def combine_track(trackFile, feature="type"):
    trackDF = pd.read_csv(trackFile)
    featureDF = find_pair(trackFile)
    trackDF[feature] = np.nan
    maxFrames = int(trackDF["Frame"].max())
    maxTracks = int(trackDF["Track_ID"].max())
    for i in range(int(maxTracks)+1):
        trackFeature = featureDF[feature].iloc[i]
        trackDF[feature].iloc[(maxFrames)*(i + 1) + i] = trackFeature
        return trackDF


# Trys to find the feature file pair for either msd_ or Traj_
# Return the pd.DataFrame of that pair if found.
def find_pair(filename):
    '''
    Trys to find the feature file pair for either msd_ or Traj_ and 
    Returns the pd.DataFrame of that pair if found.
    '''
    try:
        filename = filename.replace("msd_", "").replace("Traj_", "")
        filename = filename.split("/")
        filename[-1] = "features_" + filename[-1]
        featureFile = "/".join(filename)
        return pd.read_csv(featureFile)
    except FileNotFoundError:
        print("File pair could not be found")
