import pandas as pd

import spacy
import re
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("ru_core_news_sm/ru_core_news_sm-3.7.0")


def text_vectorizer(x):
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(x)

    feature_names = vectorizer.get_feature_names_out()
    print(type(feature_names))
    bag_of_words_df = pd.DataFrame(X.toarray(), columns=feature_names)
    return bag_of_words_df


# new features
def normalize_text(text):
    return ' '.join([token.lemma_ for token in nlp(re.sub(r'[^а-яА-Я\s]', '', text).lower())])


tr_mcc_codes = pd.read_csv('data/mcc_codes.csv', sep=';', index_col='mcc_code')
tr_types = pd.read_csv('data/trans_types.csv', sep=';', index_col='trans_type')
df = pd.concat([tr_mcc_codes.reset_index(), tr_types.reset_index()], axis=1)
df['Concatenated'] = df['mcc_description'].str.cat(df['trans_description'])
s = df[df.columns[-1]].str.cat(sep=' ')
s = normalize_text(s)
l = text_vectorizer([s]).to_numpy()
print(type(l))
print(l.shape)
print(l.tolist())
print(l[0])


