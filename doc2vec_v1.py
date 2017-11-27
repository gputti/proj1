################################################
# Date: 11/26/2017 
# this is working as of date. 
# make sure to give proper docpath
# 
################################################

from os import listdir
from os.path import isfile, exists, join
from os import walk
import gensim
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedLineDocument
from pprint import pprint  # pretty-printer
import nltk
from nltk.tokenize import word_tokenize
import re
import random

############ GLOBAL VARIABLES ############ 

docpath="./apps/asciimath/sof/docs/"
filename = "output.txt"

modeldocpath="./apps/asciimath/sof/model/"
modelfilename= modeldocpath + "doc2vec.model"

labledDocs = []
stoplist = nltk.corpus.stopwords.words('english')
                
############ START OF FUNCTIONS ############ 



def getAllFilesAsList(dirpath):
    files = []
    for (dirpath, dirnames, filenames) in walk(docpath):
        for fn in filenames:
            if not fn.startswith('.'):
                files.append(fn)
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


def getLabledDocs(f):
    documents = []
    for index, text in enumerate(open(docpath+f)):
        text = generate_new_text(text)
        words1 = [w for w in word_tokenize(text) if w not in stoplist]
        documents.append( LabeledSentence(words=words1, tags=['%s_%s' %(f,index)]) )
    return documents

def getLineFromLabledDocs(tag):
    tempfilename, index = tag.split("_")
    #print ("filename: " + filename + ", index: " + index )
    return getLineFromFile(docpath + tempfilename, index)


def debugprint(docs):
    index = 0
    print ">> total no of labled docs: ", len(labledDocs)
    for ld in labledDocs:
        index = index + 1
        if index > 30:
            break
        print ld[1], ' - ', ld[0]

def printSimOutput(sims):
    indx = len(sims)    
    print 
    print 'Total num of results : ' , indx 
    cntr = 1
    for index, label in sims:
        print cntr,') ', getLineFromLabledDocs(index)
        cntr = cntr + 1
        print ""    


def findSimilarByTakingInputline():
    new_doc=''
    while True:    # infinite loop
        new_doc = raw_input("\nEnter the question(q to quit): ")
        if new_doc == "q":
            break 
        print " "
        print 'Finding similar documents to: ' , new_doc
        print    
        sims = model.docvecs.most_similar(positive=[model.infer_vector(new_doc.split())], topn=3)
        #sims = model.docvecs.most_similar(positive=[model.infer_vector(new_doc)], topn=3)
        #sims = model.docvecs.most_similar([0], topn=3)
        
        printSimOutput(sims)

def findSimilarByRandomNumber():
    new_doc=''
    modcount = model.docvecs.count
    rand = 0
    while True:
        toquit = raw_input("\nEnter q to quit: ")        
        if toquit == "q":
            break
        rand = random.randint(0,modcount)
        text = getLineFromFile(docpath+ filename, rand)
        print ("finding similar doc for: " + text )
        sims = model.docvecs.most_similar([rand], topn=3)
        printSimOutput(sims)

############ END OF FUNCTIONS ############ 

print("")
print('started......')
print 

model = None
if isfile(modelfilename):
    print "model file exists."
    model = gensim.models.Doc2Vec.load(modelfilename)

skip = False
if model:
    print "model is loaded."
    skip = True

if skip:
    print ("skipping the building model")
else:

    files = getAllFilesAsList(docpath)
    print("Reading following files:")
    print(files)

    for f in files:
        labledDocs.extend(getLabledDocs(f))

    #debugprint(labledDocs)

    model = gensim.models.Doc2Vec(size=10, window=10, min_count=5, workers=8,alpha=0.025, min_alpha=0.025)
    model.build_vocab(labledDocs)

    for epoch in range(20):
        model.train(labledDocs,total_examples=model.corpus_count, epochs = model.iter)
        model.alpha -= 0.002 # decrease the learning rate
        model.min_alpha = model.alpha # fix the learning rate, no deca
        model.train(labledDocs,total_examples=model.corpus_count, epochs = model.iter)

    model.save(modelfilename)


modcount = model.docvecs.count
print('no of model docs: ', modcount)
modcount = modcount - 1


#findSimilarByTakingInputline()

findSimilarByRandomNumber()


print("")
print("Done!")
