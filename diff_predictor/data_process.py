import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def generate_fullstats(dataset_path, filelist, targets, target_col_name='Target'):
    """
    Generates single csv of all statatistics from list of files
    Parameters
    ---------
    dataset_path: string
        string of path to folder containing data files
    filelist: list
        list containing filenames of all files to be processed
    targets: list
        list containing strings that state which class/group a file is from,
        string must be in the filename of the data files
    Target: string
        
    Returns
    -------
    fstats_tot: pandas.DataFrame
        dataframe containing all rows from data files and with new column
        for the class/group the row came from
    """
    fstats_tot = None
    video_num = 0
    for filename in filelist:
            fstats = pd.read_csv(dataset_path + filename, encoding = "ISO-8859-1", index_col='Unnamed: 0')
            #print('{} size: {}'.format(filename, fstats.shape))
            
            for i in range(0, len(targets)):
                if targets[i] in filename:
                    print('Adding file {} size: {}'.format(filename, fstats.shape))
                    fstats[target_col_name] = pd.Series(fstats.shape[0]*[targets[i]], index=fstats.index)
                    fstats['Filename'] = pd.Series(fstats.shape[0]*[filename], index=fstats.index)
                    fstats['Video Number'] = pd.Series(fstats.shape[0]*[video_num], index=fstats.index)
                    if fstats_tot is None:
                        fstats_tot = fstats
                    else:
                        fstats_tot = fstats_tot.concat(fstats, ignore_index=True)
                    video_num += 1   
    return fstats_tot

def balance_data(df, target, **kwargs):
    """
    Balance spatial data using undersampling. Assumes input will
    be a dataframe and data will be used for categorical classification
    Parameters
    ----------
    df : pandas.DataFrame
        pandas dataframe to be balanced
    target : string
        the name of the target/tag/y-value column to balance data around
        
    Optional Parameters
    -------------------
    random_state : int : 1
        seed to base random sampling from
    Returns
    -------
    A fully balanced pandas dataframe
    """
    if 'random_state' not in kwargs:
        random_state = 1
    else:
        random_state = kwargs['random_state']
    df_target = []
    bal_df = []
    for name in df[target].unique():
        df_target.append((name, df[df[target] == name]))
    print(f"Ratio before data balance " +
          f"({':'.join([str(i[0]) for i in df_target])}) = " +
          f"{':'.join([str(len(i[1])) for i in df_target])}")
    for i in range(len(df_target)):
        ratio = min([len(i[1]) for i in df_target])/len(df_target[i][1])
        bal_df.append(df_target[i][1].sample(frac=ratio,
                                             random_state=random_state))
    print(f"Ratio after balance " +
          f"({':'.join([str(i[0]) for i in df_target])}) = " +
          f"{':'.join([str(len(i)) for i in bal_df])}")
    assert len(bal_df) > 0, 'DataFrame cant be empty'
    return pd.concat(bal_df)


def bin_data(bal_ecm, resolution=128):
    """
    Takes in a dataframe that has a binx and a biny column, and uses
    those columns to generate a bin column based on the resolution
    This is necessary for eventual cross validation to prevent data leakage

    Parameters
    ----------
    bal_ecm: pandas.DataFrame
        dataframe to be processed. Dataframe may need to have balanced classes - use balance_data function
    resolution: int
        integer representing the size of the bins. Resolution must be a factor of 2048 and > 128
        default is 128

    Returns
    -------
    bal_ecm: pandas.DataFrame
        dataframe with new column indicating which bin a give row is in
    """
    assert not 2048%resolution and resolution >= 128, "resolution needs to be a factor of 2048 and > 128"
    bins = list(range(0, 2048+1, resolution))
    bin_labels = [int(i/resolution) for i in bins][:-1]
    bal_ecm['binx'] = pd.cut(bal_ecm['X'], bins, labels=bin_labels, include_lowest=True)
    bal_ecm.loc[bal_ecm['X'] < 0] = 0
    bal_ecm['biny'] = pd.cut(bal_ecm.Y, bins, labels=bin_labels, include_lowest=True)
    bal_ecm['bins'] = (len(bins)-1)*bal_ecm['binx'].astype(np.int32) + bal_ecm['biny'].astype(np.int32)
    bal_ecm = bal_ecm[np.isfinite(bal_ecm['bins'])]
    bal_ecm['bins'] = bal_ecm['bins'].astype(int)
    return bal_ecm

def split_data(df, target, train_split, test_val_split=1.0, seed=1234):
    """
    Parameters
    ----------
    
    Returns
    -------
    
    """
    np.random.seed(seed)
    le = preprocessing.LabelEncoder()
    df['encoded_target'] = le.fit_transform(df[target])
    training_bins = np.random.choice(df.bins.unique(),
                                     int(len(df.bins.unique())*train_split),
                                     replace=False)
    X_train = df[df.bins.isin(training_bins)]
    X_test_val = df[~df.bins.isin(training_bins)]
    result = []
    if test_val_split == 1.0:
        X_test = X_test_val
    else:
        X_val, X_test = train_test_split(X_test_val,
                                         test_size=test_val_split,
                                         random_state=seed)
        y_val = X_val['encoded_target']
        result = [(X_val, y_val)]
    y_train = X_train['encoded_target']
    y_test = X_test['encoded_target']
    result = np.append([(X_train, y_train), (X_test, y_test)], result)
    return result, le
