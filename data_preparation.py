import pandas as pd
import os
from tqdm.notebook import tqdm
import spacy
import re
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("ru_core_news_sm/ru_core_news_sm-3.7.0")


def text_vectorizer(x):
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(x)

    feature_names = vectorizer.get_feature_names_out()
    bag_of_words_df = pd.DataFrame(X.toarray(), columns=feature_names)
    print(bag_of_words_df.shape)
    return bag_of_words_df


# new features
def normalize_text(text):
    return ' '.join([token.lemma_ for token in nlp(re.sub(r'[^а-яА-Я\s]', '', text).lower())])


def cool_features_creation(x):
    features = []
    features.append(
        pd.Series(x["mcc_description"].str.cat(sep=' ')) + ' ' + pd.Series(x["trans_description"].str.cat(sep=' ')))
    return pd.concat(features)


def features_creation(x):
    features = []
    features.append(pd.Series(x['day'].value_counts(normalize=True).add_prefix('day_')))
    features.append(pd.Series(x['month'].value_counts(normalize=True).add_prefix('month_')))
    features.append(pd.Series(x['hour'].value_counts(normalize=True).add_prefix('hour_')))
    features.append(pd.Series(x['night'].value_counts(normalize=True).add_prefix('night_')))
    features.append(pd.Series(x[x['amount'] > 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count']) \
                              .add_prefix('positive_transactions_')))
    features.append(pd.Series(x[x['amount'] <= 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count']) \
                              .add_prefix('negative_transactions_')))
    return pd.concat(features)


def data_prep(PATH_DATA):
    # reading data
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    tr_mcc_codes = pd.read_csv(os.path.join(PATH_DATA, 'mcc_codes.csv'), sep=';', index_col='mcc_code')
    tr_types = pd.read_csv(os.path.join(PATH_DATA, 'trans_types.csv'), sep=';', index_col='trans_type')
    transactions = pd.read_csv(os.path.join(PATH_DATA, 'transactions.csv'))
    gender_train = pd.read_csv(os.path.join(PATH_DATA, 'train.csv'), index_col='client_id')
    gender_test = pd.read_csv(os.path.join(PATH_DATA, 'test.csv'), index_col='client_id')

    tr_mcc_codes['mcc_description'] = tr_mcc_codes.mcc_description.apply(normalize_text)
    tr_types['trans_description'] = tr_types.trans_description.apply(normalize_text)

    transactions = transactions.merge(tr_mcc_codes, on='mcc_code', how='left')
    transactions = transactions.merge(tr_types, on='trans_type', how='left').set_index('client_id')

    transactions['term_id'] = 0 if transactions['term_id'].empty else 1

    transactions_train = transactions.join(gender_train, how='inner')
    transactions_test = transactions.join(gender_test, how='inner')

    for df in [transactions_train, transactions_test]:
        df['day'] = df['trans_time'].str.split().apply(lambda x: int(x[0]) % 7)
        df['month'] = df['trans_time'].str.split().apply(lambda x: int(x[0]) % 365 % 30)
        df['hour'] = df['trans_time'].apply(lambda x: re.search(' \d*', x).group(0)).astype(int)
        df['night'] = ~df['hour'].between(6, 22).astype(int)

    tqdm.pandas(desc="Progress")

    data_train = transactions_train.groupby(transactions_train.index).apply(features_creation).unstack(-1)
    data_test = transactions_test.groupby(transactions_test.index).apply(features_creation).unstack(-1)

    data_train = pd.concat(
        [data_train, transactions_train.groupby(transactions_train.index).apply(cool_features_creation)], axis=1)
    data_test = pd.concat([data_test, transactions_test.groupby(transactions_test.index).apply(cool_features_creation)],
                          axis=1)

    for df in (data_train, data_test):
        column_names = df.columns
        last_column_name = column_names[-1]
        df.rename(columns={last_column_name: 'text'}, inplace=True)

    vectorized_df = text_vectorizer(data_train['text'].values)
    data_train = pd.concat([data_train.reset_index(), vectorized_df], axis=1)
    data_test = pd.concat([data_test.reset_index(), text_vectorizer(data_test['text'].values)], axis=1)

    data_train.set_index('client_id', inplace=True)
    data_test.set_index('client_id', inplace=True)
    target = data_train.join(gender_train, how='inner')['gender']

    data_train = data_train.fillna(0).drop('text', axis=1)
    data_test = data_test.fillna(0).drop('text', axis=1)

    data_train.to_csv("data_train_temp.csv")
    data_test.to_csv("data_test_temp.csv")
    target.to_csv("target_temp.csv")
    return data_train, data_test, target

