################################################
# Date: 11/26/2017 
# this is working as of date. 
# make sure to give proper docpath
# 
################################################

import argparse
from os import listdir
from os.path import isfile, exists, join
from os import walk
import gensim
from gensim.models.doc2vec import LabeledSentence
#from gensim.models.doc2vec import TaggedLineDocument
from pprint import pprint  # pretty-printer
import nltk
from nltk.tokenize import word_tokenize
import re
import random
from random import shuffle
import sys

############ GLOBAL VARIABLES ############ 

labledDocs = []
stoplist = nltk.corpus.stopwords.words('english')
                
############ START OF FUNCTIONS ############ 



def getAllFilesAsList(dirpath):
    files = []
    print dirpath
    for (dirpath, dirnames, filenames) in walk(docpath):
        for fn in filenames:
            if not fn.startswith('.'):
                files.append(dirpath+fn)
    return files    

## line numbers start from 0
def getLineFromFile(srcfilename,linenum):
    linenum = int(linenum)
    with open(srcfilename, "r") as ins:
        index = 0
        for line in ins:
            if index == linenum:
                return line
            else:
                index += 1

def generate_new_text(text):
        no_punctuation = re.sub("[^A-Za-z0-9]", " ", text.lower())
        no_punctuation = re.sub("\s+", " ", no_punctuation)
        return no_punctuation


## provide a fully qualified path of file
def getLabledDocs(f):
    documents = []
    #print(">>> file: " + f)
    for index, text in enumerate(open(f)):
        text = generate_new_text(text)
        words1 = [w for w in word_tokenize(text) if w not in stoplist]
        documents.append( LabeledSentence(words=words1, tags=['%s_%s' %(f,index)]) )
    return documents

def getLineFromLabledDocs(file1, tag):
    tempfilename, index = tag.split("_")
    #print ("filename: " + file1 + ", index: " + index )
    return getLineFromFile(file1, index)


def debugprint(docs):
    index = 0
    print ">> total no of labled docs: ", len(labledDocs)
    for ld in labledDocs:
        index = index + 1
        if index > 30:
            break
        print ld[1], ' - ', ld[0]

def printSimOutput(file1, sims):
    indx = len(sims)    
    print 
    print 'Total num of results : ' , indx 
    cntr = 1
    for index, label in sims:
        print ">>>", index, label
        print cntr,') ', getLineFromLabledDocs(file1,index)
        cntr = cntr + 1
        print ""    


def findSimilarByTakingInputline(files):
    new_doc=''
    while True:    # infinite loop
        new_doc = raw_input("\nEnter the question(q to quit): ")
        if new_doc == "q":
            break 
        print " "
        print 'Finding similar documents to: ' , new_doc
        print    
        #sims = model.docvecs.most_similar(positive=[model.infer_vector(new_doc.split())], topn=3)
        ## below is giving better results
        sims = model.docvecs.most_similar(positive=[model.infer_vector(new_doc)], topn=3)
        printSimOutput(files[0], sims)

def findSimilarByRandomNumber(files):
    print (">> files", files)
    new_doc=''
    modcount = model.docvecs.count
    rand = 0
    while True:
        toquit = raw_input("\nEnter q to quit: ")        
        if toquit == "q":
            break
        rand = random.randint(0,modcount)
        text = getLineFromFile(files[0], rand)
        print ("finding similar doc for: " + text )
        sims = model.docvecs.most_similar([rand], topn=3)
        printSimOutput(files[0], sims)


def shudffledocs():
    shuffle(labledDocs)
    return labledDocs

############ END OF FUNCTIONS ############ 

print("")
print('started......')
print 


parser = argparse.ArgumentParser(description='Process input arguments.')
parser.add_argument('-f', default='0', type=int, dest='findby', help='0 - random docvec, 1 - you have to give doc ') 
parser.add_argument('-d', dest='docpath', type=str, default='./docs/', help='documents directory') 
parser.add_argument('-m', dest='modelfilename', type=str, default='./model/doc2vec.model', help='model file name') 


args = parser.parse_args()
findby = args.findby
docpath=args.docpath
if docpath == None:
    print "dir loc is NOT provided. So exiting....."
    print "for help of params use '-h' option."
    #sys.exit()



modelfilename=args.modelfilename

print ( "findby: " , findby )
print ( "docpath: " , docpath )
print ( "modelfilename: " , modelfilename )

model = None
if isfile(modelfilename):
    print "model file exists."
    model = gensim.models.Doc2Vec.load(modelfilename)

skip = False
if model:
    print "model is loaded."
    skip = True

files = [];

if skip:
    print ("skipping the building model")
else:
    files = getAllFilesAsList(docpath)
    print("Reading following files:")
    print(files)

    for f in files:
        labledDocs.extend(getLabledDocs(f))

    #debugprint(labledDocs)
    model = gensim.models.Doc2Vec(dm=0,size=100, window=10, min_count=2, workers=8,alpha=0.025, min_alpha=0.025)
    model.build_vocab(labledDocs)
    for epoch in range(20):
        shudffledocs()
        model.train(labledDocs,total_examples=model.corpus_count, epochs = 1)
        model.alpha -= 0.001 # decrease the learning rate
        model.min_alpha = model.alpha # fix the learning rate, no deca

    model.save(modelfilename)

modcount = model.docvecs.count
print('no of model docs: ', modcount)
modcount = modcount - 1

files = getAllFilesAsList(docpath)
if findby == 0:
    findSimilarByRandomNumber(files)
else:
    findSimilarByTakingInputline(files)


print("")
print("Done!")
