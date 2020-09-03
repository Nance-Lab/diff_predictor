import os
import boto3
import pandas as pd
import diff_classifier.aws as aws
from itertools import cycle

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
                file_data = pd.read_csv(filename, encoding = "ISO-8859-1", index_col='Unnamed: 0')
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

# Load test or traininng data
def load_input_data():

    pass

# Bin data for use in checkerboard selection
def bin_data():

    return

# Checkerboard method to avoid data bleed in prediction
def checkerboard():
    return

# Split data into training and testing data
def split_data(data, frac = 0.8, val=False, **kwargs):
    if 'checkerboard' in kwargs:
        pass
    if 'seed' in kwargs:
        seed = kwargs['seed']

    return trainx, trainy, testx, testy

# Balance out data for model input
def bal_data():
    pass
