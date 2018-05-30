# date: 5/30/2018 
# this is version-2  working model (results are below avg to avg. )
# now working on improving. 
# To run : make sure to have a file './data/tagged_plots_movielens.csv'
# Thanks to Radim. I used data from his github at
# https://github.com/RaRe-Technologies/movie-plots-by-genre/blob/master/data/tagged_plots_movielens.csv
#

import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

import seaborn as sns

import nltk
from nltk.corpus import stopwords

movie1 = "The Second Best Exotic Marigold Hotel is the expansionist dream of Sonny (Dev Patel), and it's making more claims on his time than he has available, considering his imminent marriage to the love of his life, Sunaina (Tina Desai). Sonny has his eye on a promising property now that his first venture, The Best Exotic Marigold Hotel for the Elderly and Beautiful, has only a single remaining vacancy - posing a rooming predicament for fresh arrivals Guy (Richard Gere) and Lavinia (Tamsin Greig). Evelyn and Douglas (Judi Dench and Bill Nighy) have now joined the Jaipur workforce, and are wondering where their regular dates for Chilla pancakes will lead, while Norman and Carol (Ronald Pickup and Diana Hardcastle) are negotiating the tricky waters of an exclusive relationship, as Madge (Celia Imrie) juggles two eligible and very wealthy suitors. Perhaps the only one who may know the answers is newly installed co-manager of the hotel, Muriel (Maggie Smith), the keeper of everyone's secrets. As the demands of a traditional Indian wedding threaten to engulf them all, an unexpected way forward presents itself."

print('started......')

df = pd.read_csv('data/tagged_plots_movielens.csv')
#print(df.head())

col = ['tag', 'movie_desc']
df = df[col]
df = df[pd.notnull(df['movie_desc'])]
df.columns = ['tag', 'movie_desc']

df['category_id'] = df['tag'].factorize()[0]
category_id_df = df[['tag', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'tag']].values)
#print(df.head())

'''
# draws some plot
fig = plt.figure(figsize=(8,6))
df.groupby('tag').movie_desc.count().plot.bar(ylim=0)
plt.show()
'''

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', \
		encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.movie_desc).toarray()
labels = df.category_id

X_train, X_test, y_train, y_test = train_test_split(df['movie_desc'], df['tag'], random_state = 0)
count_vect = CountVectorizer( ) 

X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)


model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = \
		train_test_split(features, labels, df.index, test_size=0.75)
model.fit(X_train, y_train)
#model.fit(features, labels)


texts = [movie1]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
for text, predicted in zip(texts, predictions):
  print('predicting for movie: ' + '"{}'.format(text)[:60] + '... "')
  print("Predicted as: '{}'".format(id_to_category[predicted]))
  print("")


'''
print("now predicting animation1 - ")
print(clf.predict(count_vect.transform([animation1])))
'''
print("now predicting scifi1 - result is: ", clf.predict(count_vect.transform([movie1]))[0])


