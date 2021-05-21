import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, scale

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

                    fstats['Video Number'] = pd.Series(fstats.shape[0]*[video_num], index=fstats.index)
                    if fstats_tot is None:
                        fstats_tot = fstats
                    else:
                        fstats_tot = fstats_tot.append(fstats, ignore_index=True)
                    video_num += 1
                    #break

            
    return fstats_tot

def balance_data(df, target, **kwargs):
    """
    Balances the dataset so there are equal number of rows for each class
    Parameters:
    ----------
    df: pandas.DataFrame
        dataframe to be balanced
    target: string
        name of dataframe column that represents that class the row is from

    Returns:
    --------
    bal_df: pandas.DataFrame
        dataframe with equal number of rows per unique class
    """
    if 'random_state' not in kwargs:
        random_state = 1
    else:
        random_state = kwargs['random_state']
    df_target = []
    bal_df = []
    for name in df[target].unique():
        df_target.append((name, df[df[target] == name]))
    print(f"Ratio before data balance ({':'.join([str(i[0]) for i in df_target])}) = {':'.join([str(len(i[1])) for i in df_target])}")
    for i in range(len(df_target)):
        ratio = min([len(i[1]) for i in df_target])/len(df_target[i][1])
        bal_df.append(df_target[i][1].sample(frac=ratio, random_state=random_state))
    print(f"Ratio after balance ({':'.join([str(i[0]) for i in df_target])}) = {':'.join([str(len(i)) for i in bal_df])}")
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
    

def scale_features(df, columns):
    """
    Scales the features in a dataframe using sklearn functions. Needed before using unsupervised learning algorithms

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe to be scaled
    columns: list
        list of column names to be to be scaled. Only use numerical columns
    
    Returns
    -------
    scaled_features: pandas.DataFrame
        dataframe of the selected features with scaled values
    """
    features_df = df[columns]
    features_df = features_df[~features_df.isin([np.nan, np.inf, -np.inf]).any(1)] # removes rows with nan or inf points
    ss = StandardScaler()
    scaled_data = pd.DataFrame(ss.fit_transform(features_df.values), columns=features_df.columns)
    scaled_data = scale(scaled_data, axis=1)
    scaled_features = pd.DataFrame(scaled_data, columns = features_df.columns)
    return scaled_features