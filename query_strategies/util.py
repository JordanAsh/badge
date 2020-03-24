import numpy as np
import pandas as pd

def save_df_as_npy(path, df):
    """
    Save pandas dataframe (multi-index or non multi-index) as an NPY file
    for later retrieval. It gets a list of input dataframe's index levels,
    column levels and underlying array data and saves it as an NPY file.

    Parameters
    ----------
    path : str
        Path for saving the dataframe.
    df : pandas dataframe
        Input dataframe's index, column and underlying array data are gathered
        in a nested list and saved as an NPY file.
        This is capable of handling multi-index dataframes.

    Returns
    -------
    out : None

    """

    if df.index.nlevels>1:
        lvls = [list(i) for i in df.index.levels]
        lbls = [list(i) for i in df.index.labels]
        indx = [lvls, lbls]
    else:
        indx = list(df.index)

    if df.columns.nlevels>1:
        lvls = [list(i) for i in df.columns.levels]
        lbls = [list(i) for i in df.columns.labels]
        cols = [lvls, lbls]
    else:
        cols = list(df.columns)

    data_flat = df.values.ravel()
    df_all = [indx, cols, data_flat]
    np.save(path, df_all)

def load_df_from_npy(path):
    """
    Load pandas dataframe (multi-index or regular one) from NPY file.

    Parameters
    ----------
    path : str
        Path to the NPY file containing the saved pandas dataframe data.

    Returns
    -------
    df : Pandas dataframe
        Pandas dataframe that's retrieved back saved earlier as an NPY file.

    """

    df_all = np.load(path)
    if isinstance(df_all[0][0], list):
        indx = pd.MultiIndex(levels=df_all[0][0], labels=df_all[0][1])
    else:
        indx = df_all[0]

    if isinstance(df_all[1][0], list):
        cols = pd.MultiIndex(levels=df_all[1][0], labels=df_all[1][1])
    else:
        cols = df_all[1]

    df0 = pd.DataFrame(index=indx, columns=cols)
    df0[:] = df_all[2].reshape(df0.shape)
    return df0

def max_columns(df0, cols=''):
    """
    Get dataframe with best configurations

    Parameters
    ----------
    df0 : pandas dataframe
        Input pandas dataframe, which could be a multi-index or a regular one.
    cols : list, optional
        List of strings that would be used as the column IDs for
        output pandas dataframe.

    Returns
    -------
    df : Pandas dataframe
        Pandas dataframe with best configurations for each row of the input
        dataframe for maximum value, where configurations refer to the column
        IDs of the input dataframe.

    """

    df = df0.reindex_axis(sorted(df0.columns), axis=1)
    if df.columns.nlevels==1:
        idx = df.values.argmax(-1)
        max_vals = df.values[range(len(idx)), idx]
        max_df = pd.DataFrame({'':df.columns[idx], 'Out':max_vals})
        max_df.index = df.index
    else:
        input_args = [list(i) for i in df.columns.levels]
        input_arg_lens = [len(i) for i in input_args]

        shp = [len(list(i)) for i in df.index.levels] + input_arg_lens
        speedups = df.values.reshape(shp)

        idx = speedups.reshape(speedups.shape[:2] + (-1,)).argmax(-1)
        argmax_idx = np.dstack((np.unravel_index(idx, input_arg_lens)))
        best_args = np.array(input_args)[np.arange(argmax_idx.shape[-1]), argmax_idx]

        N = len(input_arg_lens)
        max_df = pd.DataFrame(best_args.reshape(-1,N), index=df.index)
        max_vals = speedups.max(axis=tuple(-np.arange(len(input_arg_lens))-1)).ravel()
        max_df['Out'] = max_vals
    if cols!='':
        max_df.columns = cols
    return max_df
