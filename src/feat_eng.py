import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib


def remove_drift_train(train):
    def f_train(x, low, high, mid):
        return -((-low + high) / 625) * (x - mid) ** 2 + high - low
    a = 500000
    b = 600000
    train.loc[train.index[a:b], 'signal'] = train['signal'][a:b].values - 3 * (
            train['time'].values[a:b] - 50) / 10
    d = {7: [-1.817, 3.186, 325],
         8: [-0.094, 4.936, 375],
         9: [1.715, 6.689, 425],
         10: [3.361, 8.45, 475]}
    batches = [7, 8, 9, 10]
    for batch in batches:
        a = 500000 * (batch - 1)
        b = 500000 * batch
        values = d[batch]
        x, y, z = values[0], values[1], values[2]
        train.loc[train.index[a:b], 'signal'] = train.signal.values[a:b] - f_train(
            train.time[a:b].values, x, y, z)
    return train


def remove_drift_test(test):
    def f_test(x):
        return -(0.00788) * (x - 625) ** 2 + 2.345 + 2.58
    start = 500
    a = 0
    b = 100000
    test.loc[test.index[a:b], 'signal'] = test['signal'].values[a:b] - 3 * (
            test['time'].values[a:b] - start) / 10.
    start = 510
    a = 100000
    b = 200000
    test.loc[test.index[a:b], 'signal'] = test['signal'].values[a:b] - 3 * (
            test['time'].values[a:b] - start) / 10.
    start = 540
    a = 400000
    b = 500000
    test.loc[test.index[a:b], 'signal'] = test['signal'].values[a:b] - 3 * (
            test['time'].values[a:b] - start) / 10.

    start = 560
    a = 600000
    b = 700000
    test.loc[test.index[a:b], 'signal'] = test['signal'].values[a:b] - 3 * (
            test['time'].values[a:b] - start) / 10.
    start = 570
    a = 700000
    b = 800000
    test.loc[test.index[a:b], 'signal'] = test['signal'].values[a:b] - 3 * (
            test['time'].values[a:b] - start) / 10.
    start = 580
    a = 800000
    b = 900000
    test.loc[test.index[a:b], 'signal'] = test['signal'].values[a:b] - 3 * (
            test['time'].values[a:b] - start) / 10.

    a = 1000000
    b = 1500000
    test.loc[test.index[a:b], 'signal'] = test['signal'].values[a:b] - f_test(
        test['time'][a:b].values)
    return test


def feat_engg(df):
    batch = 50000
    shift_size = [1, 2, 3]
    add_pct_change = True
    add_pct_change_lag = True
    df = df.sort_values(by=['time']).reset_index(drop=True)
    df.index = ((df.time * 10000) - 1).values
    df['batch'] = df.index // batch
    df['batch_index'] = df.index - (df.batch * batch)
    df['batch_slices'] = df['batch_index'] // 5000
    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]),
                                   axis=1)
    if add_pct_change:
        df['pct_change'] = df['signal'].pct_change()
    for c in tqdm(['batch', 'batch_slices2']):
        d = {'mean' + c: df.groupby([c])['signal'].mean(),
             'median' + c: df.groupby([c])['signal'].median(),
             'max' + c: df.groupby([c])['signal'].max(),
             'min' + c: df.groupby([c])['signal'].min(),
             'std' + c: df.groupby([c])['signal'].std(),
             'mean_abs_chg' + c: df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x)))),
             'abs_max' + c: df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x))),
             'abs_min' + c: df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))}
        d['range' + c] = d['max' + c] - d['min' + c]
        d['maxtomin' + c] = d['max' + c] / d['min' + c]
        d['abs_avg' + c] = (d['abs_min' + c] + d['abs_max' + c]) / 2
        for v in d:
            df[v] = df[c].map(d[v].to_dict())

        # add shifts
    for shift in shift_size:
        df['signal_shift_pos_' + str(shift)] = df['signal'].shift(periods=shift)
        df['signal_shift_neg_' + str(shift)] = df['signal'].shift(periods=-1 * shift)
        for i in tqdm(df[df['batch_index'].isin(range(shift))].index):
            df['signal_shift_pos_' + str(shift)][i] = np.nan
        for i in tqdm(df[df['batch_index'].isin(range(batch - shift, batch))].index):
            df['signal_shift_neg_' + str(shift)][i] = np.nan

        if add_pct_change_lag:
            df['pct_change_shift_pos_' + str(shift)] = df['pct_change'].shift(shift)
            df['pct_change_shift_neg_' + str(shift)] = df['pct_change'].shift(-1 * shift)
            for i in df[df['batch_index'].isin(range(shift))].index:
                df['pct_change_shift_pos_' + str(shift)][i] = np.nan
            for i in df[df['batch_index'].isin(range(batch - shift, batch))].index:
                df['pct_change_shift_neg_' + str(shift)][i] = np.nan

    for c in [c1 for c1 in df.columns if
              c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_index', 'batch_slices',
                         'batch_slices2']]:
        df[c + '_msignal'] = df[c] - df['signal']
    window_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    for window in window_sizes:
        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()
        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()
        df["rolling_var_" + str(window)] = df['signal'].rolling(window=window).var()
        df["rolling_min_" + str(window)] = df['signal'].rolling(window=window).min()
        df["rolling_max_" + str(window)] = df['signal'].rolling(window=window).max()
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)
    return df


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        if col != ['open_channels', 'time']:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


class FeatureEngineering:
    def __init__(self):
        self.train = pd.read_csv('../inputs/train.csv')
        self.test = pd.read_csv('../inputs/test.csv')
        if not os.path.exists('../inputs/feature_engineered'):
            os.mkdir("../inputs/feature_engineered")

    def main(self):
        print("Removing drift from train")
        self.train = remove_drift_train(train=self.train)
        print("Removing drift from test")
        self.test = remove_drift_test(test=self.test)
        print("feature engineering train")
        self.train = feat_engg(df=self.train)
        print("Feature engineering test")
        self.test = feat_engg(df=self.test)
        print("Reducing memory size")
        self.train = reduce_mem_usage(df=self.train)
        self.test = reduce_mem_usage(df=self.test)
        print('dumping feature engineered data sets into inputs/feature_engineered')
        joblib.dump(self.train, '../inputs/feature_engineered/train.pkl')
        joblib.dump(self.test, '../inputs/feature_engineered/test.pkl')
        gc.collect()


if __name__ == '__main__':
    feature_eng = FeatureEngineering()
    feature_eng.main()
