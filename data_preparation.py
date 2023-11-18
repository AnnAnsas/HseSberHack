import pandas as pd
import os
import re
from tqdm.notebook import tqdm


# new features
def features_creation(x):
    features = []
    features.append(pd.Series(x['day'].value_counts(normalize=True).add_prefix('day_')))
    features.append(pd.Series(x['hour'].value_counts(normalize=True).add_prefix('hour_')))
    features.append(pd.Series(x['night'].value_counts(normalize=True).add_prefix('night_')))
    features.append(pd.Series(x[x['amount'] > 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
                              .add_prefix('positive_transactions_')))
    features.append(pd.Series(x[x['amount'] < 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])\
                              .add_prefix('negative_transactions_')))

    return pd.concat(features)


def data_prep(PATH_DATA):
    # reading data
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    tr_mcc_codes = pd.read_csv(os.path.join(PATH_DATA, 'mcc_codes.csv'), sep=';', index_col='mcc_code')
    tr_types = pd.read_csv(os.path.join(PATH_DATA, 'trans_types.csv'), sep=';', index_col='trans_type')
    transactions = pd.read_csv(os.path.join(PATH_DATA, 'transactions.csv'), index_col='client_id')
    gender_train = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'), index_col='client_id')
    gender_test = pd.read_csv(os.path.join(PATH_DATA, 'test.csv'), index_col='client_id')
    transactions_train = transactions.join(gender_train, how='inner')
    transactions_test = transactions.join(gender_test, how='inner')
    for df in [transactions_train, transactions_test]:
        df['day'] = df['trans_time'].str.split().apply(lambda x: int(x[0]) % 7)
        df['hour'] = df['trans_time'].apply(lambda x: re.search(' \d*', x).group(0)).astype(int)
        df['night'] = ~df['hour'].between(6, 22).astype(int)

    tqdm.pandas(desc="Progress")
    data_train = transactions_train.groupby(transactions_train.index).apply(features_creation).unstack(-1)
    data_test = transactions_test.groupby(transactions_test.index).apply(features_creation).unstack(-1)
    target = data_train.join(gender_train, how='inner')['gender']
    data_train = data_train.fillna(0)
    data_test = data_test.fillna(0)
    return data_train, data_test, target


