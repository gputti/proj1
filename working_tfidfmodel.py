################################################
# Date: 01/02/2018
# this is working as of date. 
# This uses TF-IDF model building.
# 
################################################

import sys, getopt
import tempfile
import logging
import tempfile

from os import walk
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint  # pretty-printer

import nltk
from nltk.tokenize import word_tokenize


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


############ FUNCTIONS BELOW ############ 

def getLinesArrayFromFile(filename):
    with open(filename) as f:
        #lines = f.readlines().split('.')
        ##lines = [line1 for line in f for line1 in line.split('.')]
        lines = [line1 for line1 in f]
        f.close()

    #lines = [line.rstrip('\n') for line in open(filename)]

    newlines = []
    #print (len(lines)    )
    for index, item in enumerate(lines):
        item = item.strip()
        if item and len(item) > 20:
            newlines.append(item.strip())

    lines = newlines
    #print (len(lines)    )
    #for index, item in enumerate(lines):
    #    print index, ',', len(item), ',', item[0:60]
    return lines


def getAllFilesAsList(dirpath):
    files = []
    for (dirpath, dirnames, filenames) in walk(docpath):
        for fn in filenames:
            if not fn.startswith('.'):
                files.append(fn)
    return files

def printUniqueWords(texts):
    uniwords = []
    for index, item in enumerate(texts):
        uniwords.extend([word for word in texts[index] if not word.strip() == '' ])
    print('total unique words: ', len(uniwords) )


def filterLessFrequentWords(texts, ii):
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            if not token.strip()  == '':
                frequency[token] += 1
    output = [[token for token in text if frequency[token] > ii] for text in texts]
    return output


############ END OF FUNCTIONS ############ 


meta_docpath = "meta_docs"
docpath = "docs"

documents = []

files = getAllFilesAsList(docpath)
print("reading following files:")
print(files)


for file in files:
    documents.extend(getLinesArrayFromFile(docpath+"/" +file))

print("total no of documents: ", len(documents))

#stoplist = set('for a of the and to is was in or then'.split())
stoplist = nltk.corpus.stopwords.words('english')

texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

##print ">> texts: ", len(texts), texts[999]

dictionary = corpora.Dictionary(texts)

# store the dictionary, for future reference
dictionary.save(meta_docpath + '/gopi.dict')
print  "no of words in dictionary: " , len(dictionary)


corpus = [dictionary.doc2bow(text) for text in texts]  # bow = bag of words
corpora.MmCorpus.serialize(meta_docpath + '/gopi.mm', corpus)  # store to disk, for later use

tf_idf = models.TfidfModel(corpus)
print(tf_idf)

sims = similarities.Similarity('./meta_docs/',tf_idf[corpus], num_features=len(dictionary))

print(type(sims))

new_doc = ""
while True:    # infinite loop
    new_doc = raw_input("\nEnter the question(q to quit): ")
    if new_doc == "q":
        break  # stops the loop
    query_doc = [w.lower() for w in word_tokenize(new_doc)]    
    #print(query_doc)
    query_doc_bow = dictionary.doc2bow(query_doc)
    #print(query_doc_bow)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    #print(query_doc_tf_idf)
    results = sims[query_doc_tf_idf]
    results = sorted(enumerate(results), key=lambda item: -item[1] )
    #print(results)
    for index, item in enumerate(results):
        if index < 3:
            print '#',index, ')',item[1]," - ",documents[item[0]]


print
print('DONE')
print('~')

