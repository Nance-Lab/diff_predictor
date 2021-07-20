import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from diff_predictor.core import deprecated

# if 'core' not in sys.modules:
#     from diff_predictor import core
#     import core.deprecated


def balance_data(df, target, **kwargs):
    """
    Balance spatial data using undersampling. Assumes input will 
    be a dataframe and data will be used for categorical classification
    
    Parameters
    ----------
    
    Returns
    -------
    
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

@deprecated("Old method using checkerboard, use other method")
def checkerboard(size):
    """
    Parameters
    ----------
    
    Returns
    -------
    
    """
    rows = int(size/2)
    checks = list(range(0, size*size, size+1))
    for i in range(1, rows):
        ssize = size - 2*i
        for j in range(0, ssize):
            checks.append(2*i +
                          (size+1)*j)
    for i in range(1, rows):
        ssize = size - 2*i
        for j in range(0, ssize):
            checks.append(size*size - 1 - (2*i + (size+1)*j))
    checks.sort()
    return checks


@deprecated("Old method using checkerboard, use other method")
def bin_data_checkerboard(df):
    """
    Parameters
    ----------
    
    Returns
    -------
    
    """
    bins = list(range(0, 2048+1, 256))
    df['binx'] = pd.cut(df.X, bins, labels=[0, 1, 2, 3, 4, 5, 6, 7],
                        include_lowest=True)
    df['biny'] = pd.cut(df.Y, bins, labels=[0, 1, 2, 3, 4, 5, 6, 7],
                        include_lowest=True)
    df['bins'] = 8*df['binx'].astype(np.int8) + \
        df['biny'].astype(np.int8)
    df = df[np.isfinite(df['bins'])]
    df['bins'] = df['bins'].astype(int)
    cols = df.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    df = df[cols]
    return df


def bin_data(df, res=128):
    """
    Parameters
    ----------
    
    Returns
    -------
    
    """
    assert not 2048 % res and res >= 128, \
        "resolution needs to be a factor of 2048 and > 128"
    bins = list(range(0, 2048+1, res))
    bin_labels = [int(i/res) for i in bins][:-1]
    df['binx'] = pd.cut(df.X, bins, labels=bin_labels,
                        include_lowest=True)
    df['biny'] = pd.cut(df.Y, bins, labels=bin_labels,
                        include_lowest=True)
    df['bins'] = (len(bins)-1)*df['binx'].astype(np.int32) + \
        df['biny'].astype(np.int32)
    df = df[np.isfinite(df['bins'])]
    df['bins'] = df['bins'].astype(int)
    return df


@deprecated("Old method using checkerboard, use other method")
def checkerboard_split(df, target, test_val_split=1.0, seed=1234):
    """
    Parameters
    ----------
    
    Returns
    -------
    
    """
    np.random.seed(seed)
    le = preprocessing.LabelEncoder()
    df['encoded_target'] = le.fit_transform(df[target])
    X_train = \
        df[~df.bins.isin(checkerboard(len(df.bins.unique())-1))].reset_index()
    X_test_val = \
        df[df.bins.isin(checkerboard(len(df.bins.unique())-1))].reset_index()
    result = []
    if test_val_split == 1.0:
        X_test = X_test_val
    else:
        X_val, X_test = train_test_split(X_test_val, test_size=test_val_split,
                                         random_state=seed)
        y_val = X_val['encoded_target']
        result = [(X_val, y_val)]
    y_train = X_train['encoded_target']
    y_test = X_test['encoded_target']
    result = np.append([(X_train, y_train), (X_test, y_test)], result)
    return result


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


def get_lengths(df, X_train, X_test, X_val=None):
    """
    Parameters
    ----------
    
    Returns
    -------
    
    """
    print(f'Tot before split: {len(df)}')
    print(f'Training: {len(X_train)} ({len(X_train)/len(df):.3f}%)')
    print(f'Testing: {len(X_test)} ({len(X_test)/len(df):.3f}%)')
    try:
        print(f'Evaluation: {len(X_val)} ({len(X_val)/len(df):.3f}%)')
    except Exception:
        pass
