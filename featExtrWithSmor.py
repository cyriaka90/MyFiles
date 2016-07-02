 # -*- coding: utf-8 -*- 

## This code extracts the features for several glosses and stores it in two text files to be fed to evaluation. py or predictGoodness.py

## import everything needed
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from nltk.parse.stanford import StanfordParser
from nltk.tag.stanford import StanfordPOSTagger
import string
from pattern.de import singularize
import subprocess
import os

## set variables
parser=StanfordParser(model_path="edu/stanford/nlp/models/lexparser/germanPCFG.ser.gz")
st = StanfordPOSTagger('german-dewac.tagger')
featuresPhrases = []
finalRatings = []
count=0
path = '/home/hanna/Documents/SMOR/'

## read in the word frequencies from DeReWo
derewo = open('derewo-v-ww-bll-320000g-2012-12-31-1.0.txt')
freqWo= []
freqNo= []
for lines in derewo:
	lines = lines.strip()               
	parts = lines.split(" ")
	freqWo.append(parts[0].lower())
	freqNo.append(int(float(parts[1])))

## read in the word vectors
vectors = open('decow14ax_all_min_100_vectors_50dim_l2norm_axis01_random.emb')

vecWords = []
for line in vectors:
        line = line.strip()
        numpy_arrays = []
        parts = line.split(" ")
        vecWords.append(parts[0])

vectors.seek(0)

#read in the stopword list
stop = [ ]

for line in open('stopword-list.txt'):
	line = line.strip() 
	stop.append(line)

#read in the glosses and their ratings
for lines in open('dummy.txt'):
        lines = lines.strip()               
        parts = lines.split("\t")
        count=count+1
        rating=parts[0]
        phrase=parts[1]
        if rating=="1" or rating=="0":
                finalRatings.append("1")
        elif rating=="2" or rating=="3":
                finalRatings.append("2")

        #extract the features for the classifier
        print("Extracting the features for the Classifier...")
        print(count)
        target1= " "
        sent1= phrase
        ## split sentence into words
        vec1 = []
        vec2 = []
        class_arrays = []
        wordLenMean=0   ##1
        comp=0          ##2
        adj=0		##3
        pronouns=0      ##4
        complexity=-1   ##5
        posT=0 		##6
        rareFreq = 0    ##7
        wordLenMean=0   ##8
        junk=0		##9
        cos=0		##10

        ##print("Classifiying...")
        ## split sentence into words
        wordsSplit = sent1.split(" ")

        ## Feature 1:  Sentence Length
        length=len(wordsSplit)
        class_arrays.append(length)

        for f in range(0,length):
                wordsClean=wordsSplit[f]
                if "&_" in wordsClean:
                        target1=wordsClean.translate(string.maketrans("",""), string.punctuation)
                        wordsSplit[f]=wordsClean
                        break

        ## choose tagger and tag sentence
        sentClean=str(wordsSplit)
        sentTagged=st.tag(sentClean)


        ## Feature 2: Completeness (capital word initial pos, punct. mark final)
        if wordsSplit[0][0].isupper() and (sent1.endswith(".") or sent1.endswith("!") or sent1.endswith("?")) :
                comp=1
        class_arrays.append(comp)        

        ## Feaure 5: Complexity (Stanford): how deeply embedded is the sentence?
        ##parse sentence with the Stanford parser
        parse=list(parser.raw_parse(sentClean.decode("utf-8")))
        sentParse=str(parse).split(" ")
        for i in range(0, len(sentParse)):
                if "Tree('S'" in sentParse[i]:
                        complexity=complexity+1

        class_arrays.append(complexity)

        ## Feature 6: position of target word - is it at the end?
        sentWOPunc=sentClean.translate(string.maketrans("",""), string.punctuation)
        if sentWOPunc.endswith(target1):
                posT=1

        class_arrays.append(posT)

        ## Make sentence without punctuation and target word lower case 
        sentWOPunc = sentWOPunc.lower()
        target1 = target1.lower()
        wordsLowWOPunc = sentWOPunc.split(" ")

        ##  use base forms (singular etc.)
        for j in range(0,length):

        ## Feature 3: Known word (SMOR) 
                word2 = wordsWOPunc[j]
                ps = subprocess.Popen(('echo', word2), stdout=subprocess.PIPE)
                output = subprocess.check_output([os.path.join(path, 'smor-infl')], shell=False, stdin=ps.stdout)
                ps.wait()
                print(output)
                if "no result for" in output:
                        knownWord=0
        ## Feature 4: no free pronouns (Stanford) yes(1)/no(0)
        ## go through sentence and see if there are tags for pronouns 
                wordsPOS0=sentTagged[j]
                wordsPOS=wordsPOS0[1]
                if wordsPOS=="PPOS" or wordsPOS=="PDS" or wordsPOS=="PRELS" or wordsPOS=="PWS" or wordsPOS=="PIS" or wordsPOS=="PRF" or wordsPOS=="PPER":
                        pronouns=1
                elif wordsPOS=="ADJA" or wordsPOS=="ADJD":
                        adj=adj+1

        ## Feature 7: word frequencies (look up frequencies in DeReWo)
                rf=0
                countInList=0
                word2 = wordsLowWOPunc[j]
                word2=singularize(word2.decode("utf-8")).lower()
                for i in range(0,len(freqWo)):
                        if freqWo[i].decode("utf-8")== word2:
                                freqNo[i]=rf
                                countInList=countInList+1
                                if rf > 14:
                                        rareFreq=rareFreq+1
                                        break
                ## if the word is not in the list it must be rare therefore increase count
                if countInList==0:
                        rareFreq=rareFreq+1

        ## Feat. 8: mean word length
                wordLen=len(word2)
                wordLenMean=wordLen+wordLenMean

        wordLenMean=wordLenMean/length

        ##write everything in the array
        class_arrays.append(rareFreq)
        class_arrays.append(pronouns)
        #class_arrays.append(knownWord)
        class_arrays.append(adj)
        class_arrays.append(wordLenMean)


        ## Feature 9: junk sentences - many punctuation marks (>3)? yes(1)/no (0)
        num=0
        for char in sent1:
                if char in string.punctuation:
                        num=num+1
        if num>7:
                junk=1

        class_arrays.append(junk)

        ## filter stopwords for similarity judgment

        wordsLowSt = [word for word in wordsLowWOPunc if word not in stop]

        ## Feature 10:  Semantic Similarity
        ## vector of the target word
        vecnum=0
        vone=0   
        numpy_arrays = []
        for i in range(0, len(vecWords)):
                vecnum=vecnum+1
                if target1 == vecWords[i]:
                        for line in vectors:			
                                vone=vone+1
                                if vone==vecnum:
					line = line.strip()
                                        numpy_arrays = []
                                        parts = line.split(" ")
                                        for x in range(1, 50):
                                                numpy_arrays.append(float(parts[x]))
                                        vec1=numpy_arrays
                                        break
        vectors.seek(0)

        ## vectors of the other words in the gloss
        vectot=0

        for j in range(0, len(wordsLowSt)):
                word2 = wordsLowSt[j]
                vtwo=0
                vecnum2=0
                ##10
                if word2!= target1:
                        match=0
                        for i in range(0,len(vecWords)):
                                vecnum2=vecnum2+1
                                if word2 == vecWords[i]:
                                        for line in vectors:
                                                ##line=line.strip()
                                                vtwo=vtwo+1
                                                if vtwo==vecnum2:
                                                        match=1
                                                        line = line.strip()
                                                        numpy_arrays = []
                                                        parts = line.split(" ")
                                                        for x in range(1, 50):
                                                                numpy_arrays.append(float(parts[x]))
                                                        vec2=numpy_arrays
							break
                                        vectors.seek(0)		
                                        break
                        if match==0:
                                vec2=[ ]
                        elif vec1!=[ ] and vec2!=[ ]:
                                result = (1-spatial.distance.cosine(vec1, vec2))
                                cos=cos+result
                                vectot=vectot+1
        if vectot != 0:        
                cos=cos/vectot

        class_arrays.append(cos)

	featuresPhrases.append(class_arrays)

## train the classifier

from sklearn import svm
X = featuresPhrases
y = finalRatings
f = open('Xtrain.txt','w')
f.write(str(X))
e = open('ytrain.txt','w')
e.write(str(y))
print("done")

