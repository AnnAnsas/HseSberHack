import pandas as pd

import spacy
import re
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("ru_core_news_sm")


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


tr_mcc_codes = pd.read_csv('mcc_codes.csv', sep=';', index_col='mcc_code')
tr_types = pd.read_csv('trans_types.csv', sep=';', index_col='trans_type')
print(tr_mcc_codes.head())
print(tr_types.head())
