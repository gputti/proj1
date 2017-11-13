from os import listdir
from os.path import isfile, exists, join
from os import walk
import gensim
import random
from collections import namedtuple
import DocIterator as DocIt
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedLineDocument
from pprint import pprint  # pretty-printer
import DocIterator as DocIt

############ START OF FUNCTIONS ############ 

def getLinesArrayFromFile(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
        #lines = [line1 for line in f for line1 in line.split('.')]
        f.close()

    #lines = [line.rstrip('\n') for line in open(filename)]
    print("total lines read: " + str(len(lines)))
    newlines = []
    #print (len(lines)    )
    for index, item in enumerate(lines):
        item = item.strip()
        if item and len(item) > 36:
            newlines.append(item.strip())

    lines = newlines
    print("Total lines after filtering : " + str(len(lines)) )
    #for index, item in enumerate(lines):
    #    print index, ',', len(item), ',', item[0:60
    return newlines
    #return lines[0:2]


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
        uniwords.extend([word for word in texts[index]])
    print('total unique words: ', len(uniwords) )

def getLine(linenum):
    with open("./docs/alldata-id.txt", "r") as ins:
        index = 0
        for line in ins:
            if index == linenum:
                return line
            else:
                index += 1


############ END OF FUNCTIONS ############ 

print("")
print('started......')
docpath="./docs"

modelfilename="doc2vec.model"
model = None
if isfile(modelfilename):
    model = gensim.models.Doc2Vec.load(modelfilename)

skip = False
if model:
    skip = True

if skip:
    print ("skipping the building model")
else:

    files = getAllFilesAsList(docpath)
    print("Reading following files:")
    print(files)

    raw_data=[]
    data=[]
    for file in files:
        data.extend(getLinesArrayFromFile(docpath+"/" +file))

    it = DocIt.LabeledLineSentence(files)

    model = gensim.models.Doc2Vec(size=500, window=10, min_count=2, workers=8,alpha=0.025, min_alpha=0.025) # use fixed learning rate
    model.build_vocab(it)

    for epoch in range(20):
        model.train(it,total_examples=model.corpus_count, epochs = model.iter)
        model.alpha -= 0.002 # decrease the learning rate
        model.min_alpha = model.alpha # fix the learning rate, no deca
        model.train(it,total_examples=model.corpus_count, epochs = model.iter)


modcount = model.docvecs.count
print('no of model docs: ', modcount)
modcount = modcount - 1

model.save(modelfilename)


new_doc=''
while True:    # infinite loop
    new_doc = raw_input("\nEnter the question(q to quit): ")
    if new_doc == "q":
        break  # stops the loop
    pprint("finding similar line to: " + new_doc)    
    #inferred_docvec = model.infer_vector(new_doc.split())
    #sims = model.docvecs.most_similar([inferred_docvec], topn=3)
    sims = model.docvecs.most_similar(positive=[model.infer_vector(new_doc.split())], topn=5)
    #pprint (sims)
    indx = len(sims)
    print('>>> ' , indx )
    for index, label in sims:
        #if indx == 0:
        #    break
        print(index, label, getLine(index) )
        #if data[index]:
        #    print(index , " - ",  data[index] )
        #indx -= 1

print("")
print("Done!")

