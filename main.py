from __future__ import print_function	# For Py2/3 compatibility
from nltk.stem import WordNetLemmatizer 
import math  
from nltk.stem.porter import *
import nltk
import json
import statistics 
import eel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random
import  copy
import glob
import os
# nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer() 
Ltoken = []
Lposting = []


######Vctor Generater#########     O(n1+n2)
def generate( ref_words, doc_freq, tokens, tf, i,n,y1):  #Genrates Vector
    # print(list(zip(ref_words,y1)))
    # exit()
    n1 = len(ref_words)               #Vector Length I equal to total no of terms
    n2 = len(tokens)                    #Terms length in current docs
    dot = 0
    j = 0
    k = 0
    
    vector = []
    while(k < n2 and j < n1 and tokens[k] != ''):   #Make vector untill terms in current doc ends 
        if(ref_words[j] != tokens[k]):
            if(y1[j]>2):
                vector.append(0)
            j += 1
        else:
            ###########################
            if(int(y1[j])>2):
                weight = (float(tf[k])/(len(tf)))*math.log10((n)/int(doc_freq[j]))
                dot += pow(weight, 2)
                vector.append(weight)
                                        #Calculation of tf^2 for getting Normalization
            j += 1
            k += 1
            
          
    while(j < n1):                                         #cat remaining term weights as zero
        if(ref_words[j] == ''):
            j += 1
            continue
        if(y1[j]>2):
            vector.append(0)
        j += 1
    while(k < n2):
        print(tokens)
        if(tokens[k] == ''):
            k += 1
            continue
        k += 1
    return dot,vector

#############Vector GEnerator  O(56)#####################################
def generate_vector(all_vocab,all_tf,doc_vocab,doc_tf,folder,y1):

    n = len(doc_vocab)
    
    ref_words = all_vocab
    doc_freq = all_tf
    weights = []
    vectors = {}
    for i,value in doc_vocab.items():
        tokens = doc_vocab[i]
        tf = doc_tf[i]
        dot,vector = generate(ref_words, doc_freq, tokens, tf, i ,n,y1)
        vectors[str(i)] = vector
        weights.append(math.sqrt(dot))

    file = open('./{}/Corpus.txt'.format(folder),'w')
    file.write(",".join(map(str, all_vocab)))
    file.close()
    file = open('./{}/allTermFreq.txt'.format(folder),'w')
    file.write(",".join(map(str, y1)))
    file.close()
    file = open('./{}/DocumentFrequency.txt'.format(folder),'w')
    file.write(",".join(map(str, all_tf)))
    file.close()
    file = open('./{}/Weights.txt'.format(folder),'w')
    file.write(",".join(map(str, weights)))
    file.close()
    file  = open('./{}/vector.json'.format(folder),'w')
    json.dump(vectors,file)
    file.close()
    
######################mergesort  O(n)###################################


def merge( arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m

    L = [0] * (n1)
    R = [0] * (n2)

# Copy data to temp arrays L[] and R[]
    for i in range(0, n1):
        L[i] = arr[l + i]
    for j in range(0, n2):
        R[j] = arr[m + 1 + j]
# Merge the temp arrays back into arr[l..r]
    i = 0     # Initial index of first subarray
    j = 0     # Initial index of second subarray
    k = l     # Initial index of merged subarray

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def mergeSort( arr, l, r):
    if l < r:
        m = int((l+(r-1))/2)
        mergeSort(arr, l, m)
        mergeSort(arr, m+1, r)
        merge(arr, l, m, r)

def mergelists( x1, x2,y1,y2, ind, doc_freq):
    n1 = len(x1)
    n2 = len(x2)
    word = []
    tf = []
    freq_list = []
# Merge the temp arrays back into arr[l..r]
    i = 0     # Initial index of first subarray
    j = 0     # Initial index of second subarray
    k = 0     # Initial index of merged subarray

#word->dictionary, Plist->PostingList, posting-> <DocumentNumber>position of word
    while i < n1 and j < n2:
        if(x1[i] == ''):
            i += 1
            continue
        if(x2[j] == ''):
            j += 1
            continue

        if x1[i] < x2[j]:
            word.append(x1[i])  # Send Word To Dictionary
            tf.append(y1[i])
            freq_list.append(doc_freq[i])  # Append Previous Frequency
            i += 1
        elif x1[i] > x2[j]:  # Attach Document and Position since Second list's doc is smaller
            word.append(x2[j])
            tf.append(y2[j])
            freq_list.append(1)  # New Word Append 1 as frequency
            j += 1
        else:

            # Documewnts are same so join <docNo> and both postion Lists
            word.append(x1[i])
            tf.append(y1[i]+y2[j])
            freq_list.append(doc_freq[i]+1)  # Join Previous Frequency + 1
            i += 1
            j += 1

# Copy the remaining elements of list1, if there
# are any
    while i < n1:
        if(x1[i] == ''):
            i += 1
            continue
        word.append(x1[i])
        tf.append(y1[i])
        freq_list.append(doc_freq[i])
        i += 1

# Copy the remaining elements of list2, if there
# are any
    while j < n2:
        if(x2[j] == ''):
            j += 1
            continue
        word.append(x2[j])
        tf.append(y2[j])
        freq_list.append(1)
        j += 1
    return word, freq_list,tf
def tokenizer(files,stopwords,Class,doc_vocab,doc_tf,num):
    global Ltoken
#Tokenize stem fold case of words
    
    for i in files:
        print("Toeknizing Doc {}".format(i))
        f = open(i,'rb')
        if f.mode == 'rb':
            
            content = str(f.read())
            
            w = ''
            for j in range(len(content)):

                if((content[j] in [' ', '.', '\n',']','-','?']) and w != '' and w != "'" or (content[j-1]>='a' and content[j-1]<='z' and (content[j]>'z' or content[j]<'a'))):
                    # removing stopwords
                    if(w not in stopwords and w not in ['']):
                        tk = stemmer.stem(w)
                        #tk = lemmatizer.lemmatize(w)  # Lemmatization
                        Ltoken.append(tk)

                    w = ''

                elif content[j] not in ['', ' ', '[', ',', ':', '?', '(',')','—','"',';',"'",'!','-','.','\n','']:
                    if(content[j] >= 'A' and content[j] <= 'Z'):  # Case folding
                        w = w+chr(ord(content[j])+32)
                    else:
                        w += content[j]

        # Sorting and adding frequency of Tokens In file
        print("Sorting Tokens Withing Document")
        mergeSort(Ltoken, 0, len(Ltoken)-1)
        
       
        
        
        counter = 1
        # Write token and tf to if no preceding word is same as current
        print("Removing Duplicate and Adding termfrequency")
        words = []
        tf = []
        for l in range(0, len(Ltoken)-1):
            if Ltoken[l] != Ltoken[l+1]:

                words.append(Ltoken[l])
                tf.append(counter)
                counter = 1
            else:
                counter += 1  # preceding word is same increase tf
        words.append(Ltoken[len(Ltoken)-1])
        tf.append(counter)
        doc_vocab[str(Class)+'.'+str(num)] = words
        doc_tf[str(Class)+'.'+str(num)] = tf
        num+=1
        Ltoken.clear()
       
#Document as a BLOCK sorting done
    
    return num
def Processor(doc_vocab,doc_tf,folder):
    print("IN fucntion Processor")
        
    x1 = doc_vocab[list(doc_vocab.keys())[0]]
    y1 = doc_tf[list(doc_vocab.keys())[0]]
    doc_freq = [1]*len(x1)
    count = 1
    for i,value in doc_vocab.items():                                  #Takes tokens and respective term frequency from each token file and 
        if(count == 1):
            count+=1
            continue
        x2 = doc_vocab[i]
        y2 = doc_tf[i]
        x1, doc_freq,y1 = mergelists(x1, x2,y1,y2, i, doc_freq)
    all_vocab = []
         #Writing Doc Frequency

    for i in range(len(x1)):
        all_vocab.append(x1[i])
    all_tf = []
    for i in range(len(doc_freq)):
        all_tf.append(doc_freq[i])
    generate_vector(all_vocab,all_tf,doc_vocab,doc_tf,folder,y1)    





#############Merge sorted terms of each doc ############
def merge_doc( arr, pos, l, m, r):
    n1 = m - l + 1
    n2 = r - m

    L = [0] * (n1)
    R = [0] * (n2)
    LP = [0] * (n1)
    RP = [0] * (n2)
# Copy data to temp arrays L[] and R[]
    for i in range(0, n1):
        L[i] = arr[l + i]
        LP[i] = pos[l + i]
    for j in range(0, n2):
        R[j] = arr[m + 1 + j]
        RP[j] = pos[m + 1 + j]

# Merge the temp arrays back into arr[l..r]
    i = 0     # Initial index of first subarray
    j = 0     # Initial index of second subarray
    k = l     # Initial index of merged subarray

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            pos[k] = LP[i]
            i += 1
        else:
            arr[k] = R[j]
            pos[k] = RP[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        pos[k] = LP[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        pos[k] = RP[j]
        j += 1
        k += 1

def sort_doc( arr, pos, l, r):        #######Sort tokenized terms of documents
    if l < r:
        m = int((l+(r-1))/2)
        sort_doc(arr, pos, l, m)
        sort_doc(arr, pos, m+1, r)
        merge_doc(arr, pos, l, m, r)

def parse_Query( query):
    stopwords = []
    s = open('Stopword-List.txt', 'r')
    stopdoc = s.read()
    w = ''
    print("Removing StopWords")
    for i in range(len(stopdoc)):  # Get Stopwords to remove from query
        if(stopdoc[i] == '\n'):
            if(w != ''):
                stopwords.append(w)
            w = ''
        elif stopdoc[i] != ' ':
            w += stopdoc[i]
    s.close()
    parsed = []
    w = ''
    for i in range(len(query)):  # parse Query
        if(query[i] == ' '):
            if(w not in ['', ",", ' ']):
                parsed.append(w)
            w = ''
        elif query[i] not in [' ', "'", '"', '']:
            w += query[i]
    s.close()
    if w not in ['', ",", ' ']:
        parsed.append(w)
    for i in range(len(parsed)-1, -1, -1):
        if(parsed[i] in stopwords):   #######Removing Stop Words #######
            print("{} removed".format(parsed[i]))
            parsed.pop(i)
        else:
            parsed[i] = parsed[i].lower()
            parsed[i] = stemmer.stem(parsed[i])
            #parsed[i] = lemmatizer.lemmatize(parsed[i])
    print("Unsorted Query =  {}".format(parsed))
    parsed.sort()
    print("Sorted Query = {}".format(parsed))
    return parsed




def binarySearch( arr, l, r, x):   ###########Search Words IN query through Binary searcg    Log(n)

    while l <= r:

        mid = int(l + (r - l)/2)

    # Check if x is present at mid
        if arr[mid] == x:
            return mid

    # If x is greater, ignore left half
        elif arr[mid] < x:
            l = mid + 1

    # If x is smaller, ignore right half
        else:
            r = mid - 1

    # If we reach here, then the element was not present
    return -1

def process_Query( parsed,corpus):
    tokens = corpus
    count = 1
    new_parsed = []
    query_tf = []
    for i in range(len(parsed)-1):                  #if terns in query is redundant just add 1 to count
        if(parsed[i] == parsed[i+1]):
            count += 1
        else:
            new_parsed.append(parsed[i])            #append single term and tf 
            query_tf.append(count)
            count = 1
    new_parsed.append(parsed[len(parsed)-1])
    query_tf.append(count)
    result = []
    for i in range(len(new_parsed)):               #Find occurence of query terms in corpus
        result.append(binarySearch(
            tokens, 0, len(tokens)-1, parsed[i]))

    return result, query_tf, new_parsed


def uc_distance(query,corpus):
    ans = 0
    for key,value in corpus:
        try:
            ans+= (value-query[key])**2
        except:
            ans+= value**2
    return math.sqrt(ans)
    

def fetch_docs( parsed, query_tf, results,all_vectors,weights,documentFreq,corpus,all_tf):
    n = len(all_vectors)
    #print(results)
    query_vector = []
    doc_freq = documentFreq
    ########
    weight = []
    print("Genrating Vector From Query")
    for i in range(len(results)-1,-1,-1):     #Making Vector from query term tf
        print("Document Frequency of {} = {}".format(
            parsed[i], doc_freq[results[i]]))
        
        if(int(all_tf[results[i]])>2):
            query_vector.append( (math.log10(n/int(doc_freq[results[i]])))* (query_tf[i]/len(query_tf)) )
            print("{}-->{}".format(parsed[i],(math.log10(int(doc_freq[results[i]]))/56)*query_tf[i]))
        else:
            results.pop(i)

    print("------------")
    print("Query Vector")
    print(query_vector)
    print("------------")
    normal_vectors = weights
    vectors = all_vectors
    itr = 0

    

    for i,vector in vectors.items():                                                         #Vector addition of term in query with terms in each doc. Only weights 
        vector = vectors[str(i)]
    #     weight.append(uc_distance(zip(parsed,query_vector),zip(corpus,vector)))
    # print(weight)
        ans = 0
        a = 0
        print("Vector For Doc {}".format(i))
        print("<",end='')
        for j in range(len(query_vector)):
            try:
                print("{},".format(vector[results[j]]),end=' ')
                ans = ans+ query_vector[j]*float(vector[results[j]])
                a += pow(query_vector[j], 2)
            except:
                pass
        print(">")
        b = float(normal_vectors[itr])
        itr+=1
        try:
            #rslt = math.sqrt(ans)
            rslt = round(ans/(math.sqrt(a)*b), 10)
            print("<A><B>/|A||B| = {}/ |{}||{}|".format(ans,math.sqrt(a),b))
        except:
            rslt = 0
        weight.append(rslt)

    return weight

class KNN():
    def __init__(self):
        try:
        
            file=open('./KNN/Corpus.txt','r')
            self.corpus = file.read().split(',')
            file.close()

            file=open('./KNN/DocumentFrequency.txt','r')
            self.documentFreq = file.read().split(',')
            file.close()

            file=open('./KNN/Weights.txt','r')
            self.weights = file.read().split(',')
            file.close()

            file=open('./KNN/vector.json','r')
            self.vectors = json.load(file)
            file.close()

            file = open('./KNN/allTermFreq.txt','r')
            self.all_tf = file.read().split(',')
            file.close()
        except:
            self.corpus = None
            self.documentFreq = None
            self.weights = None
            self.vectors = None
            self.all_tf = None
    def train_KNN(self):
        test = {}
        doc_vocab = {}
        doc_tf = {}
        num = 1
        P1 = os.listdir('./bbcsport-fulltext/bbcsport/')
        stopwords = []
        s = open('Stopword-List1.txt', 'r')  # picking stopwords
        stopdoc = s.read()
        w = ''
        for i in range(len(stopdoc)):
            if(stopdoc[i] == '\n'):
                if(w != ''):
                    stopwords.append(w)  # parsing stopwords
                w = ''
            elif stopdoc[i] != ' ':
                w += stopdoc[i]
        s.close()
        num2 = 1
        for p in P1:  
            path = './bbcsport-fulltext/bbcsport/{}/*.txt'.format(p)   
            files=glob.glob(path)
            if(len(files)>0):
                ratio = int(len(files)*0.3)
                train = files[ratio:]
                test[p] = files[:ratio]
                print("Length of Test and Train {},{}".format(len(train),len(test)))
                num=tokenizer(train,stopwords,p,doc_vocab,doc_tf,num)   

        Processor(doc_vocab,doc_tf,"KNN")
        print("Length of Test Data = {}".format(len(test)))
        file = open('./KNN/testDocs.txt','w')
        json.dump(test,file)
        file.close()


    def run_KNN(self):
        if self.corpus==None or self.documentFreq==None or self.weights==None or self.vectors==None:
            eel.say_hello_js("KNN Not Trained Yet!! Please Train First")
            return
        file = open('./KNN/testDocs.txt','r')
        test = json.load(file)
        file.close()

        prediction = []
        test_label = []

        print("Running KNN")
        for label,arr in test.items():
            for filename in arr:
                test_label.append(label)    
                file = open(filename,'r')
                text = file.read()
                file.close()
                parsed = parse_Query(text)
                results,query_tf,parsed = process_Query(parsed,self.corpus)
                for i in range(len(results)-1,-1,-1): #Remove words from query which are not in corpus
                    if(results[i]==-1):
                        parsed.pop(i)
                        results.pop(i)
                        query_tf.pop(i)
                result = fetch_docs(parsed,query_tf,results,self.vectors,self.weights,self.documentFreq,self.corpus,self.all_tf)
                docs = []
                for i, arr in self.vectors.items():
                    docs.append(i)
                sort_doc(result,docs,0,len(result)-1)
                p = []
                for i in range(len(docs)-1,len(docs)-4,-1):
                    p.append(docs[i].split('.')[0])
                try:
                    prediction.append(statistics.mode(p))
                except:
                    prediction.append(p[len(p)-1])
        return accuracy_score(test_label,prediction),confusion_matrix(test_label,prediction)

    def predict_text(self,text):
        if self.corpus==None or self.documentFreq==None or self.weights==None or self.vectors==None:
            eel.say_hello_js("KNN Not Trained Yet!! Please Train First")
            return
        prediction = ""
        parsed = parse_Query(text)
        results,query_tf,parsed = process_Query(parsed,self.corpus)
        for i in range(len(results)-1,-1,-1): #Remove words from query which are not in corpus
            if(results[i]==-1):
                parsed.pop(i)
                results.pop(i)
                query_tf.pop(i)
        result = fetch_docs(parsed,query_tf,results,self.vectors,self.weights,self.documentFreq,self.corpus,self.all_tf)
        docs = []
        for i, arr in self.vectors.items():
            docs.append(i)
        sort_doc(result,docs,0,len(result)-1)
        p = []
        for i in range(len(docs)-1,len(docs)-4,-1):
            p.append(docs[i].split('.')[0])
        try:
            prediction = statistics.mode(p)
        except:
            prediction = p[len(p)-1]
        return prediction
    
    




class Kmeans():


    def train_Kmean(self):
        test = {}
        doc_vocab = {}
        doc_tf = {}
        num = 1
        P1 = os.listdir('./bbcsport-fulltext/bbcsport/')
        stopwords = []
        s = open('Stopword-List.txt', 'r')  # picking stopwords
        stopdoc = s.read()
        w = ''
        for i in range(len(stopdoc)):
            if(stopdoc[i] == '\n'):
                if(w != ''):
                    stopwords.append(w)  # parsing stopwords
                w = ''
            elif stopdoc[i] != ' ':
                w += stopdoc[i]
        s.close()
        num2 = 1
        for p in P1:  
            path = './bbcsport-fulltext/bbcsport/{}/*.txt'.format(p)   
            files=glob.glob(path)
            if(len(files)>0):
                num=tokenizer(files,stopwords,p,doc_vocab,doc_tf,num)   

        Processor(doc_vocab,doc_tf,"Kmean")
        print("Length of Test Data = {}".format(len(test)))
        return test


    def uclidean_distance(self,centroids,vector,key,cluster):
        result = {}
        ans1 = ans2 = ans3 = ans4 = ans5 = 0
        for j in range(len(centroids[0])):
            ans1 += pow( (centroids[0][j] - vector[j]) , 2 )
            ans2 += pow( (centroids[1][j] - vector[j]) , 2 )
            ans3 += pow( (centroids[2][j] - vector[j]) , 2 )
            ans4 += pow( (centroids[3][j] - vector[j]) , 2 )
            ans5 += pow( (centroids[4][j] - vector[j]) , 2 )
        result['c1'] = math.sqrt(ans1)
        result['c2'] = math.sqrt(ans2)
        result['c3'] = math.sqrt(ans3)
        result['c4'] = math.sqrt(ans4)
        result['c5'] = math.sqrt(ans5)
        result  = sorted(result.items(), key=lambda x: x[1])
        result = dict(result)
        print(result)
        cluster[list(result.keys())[0]].append(key)


    

    def calculate_centroid(self,value1,vectors):
        try:
            ans = vectors[value1[0]]
        except:
            return
        
        n = len(value1)
        for i in range(1,n):
            new = vectors[value1[i]]
            for j in range(len(ans)):
                ans[j]+=new[j]
        ans = [a/n for a in ans]
        return ans

    def run_K_mean(self,max_iter = 5):
        text = ""
        try:
            file = open('./Kmean/vector.json','r')
            vectors = json.load(file)
            file.close()
        except:
            eel.say_hello_js1("Kmeans Vectors Not Found. Generate Vectors First")
            return    
        clist = []
        centroids = []
        for i in range(5):
            num = random.randint(0,len(vectors))
            vec = list(vectors.keys())[num]
            if vec in clist:
                i-=1
                continue
            else:
                clist.append(vec)
                print(vec)
                centroids.append(vectors[vec])
        old_cluster = {}
        old_cluster['c1'] = []
        old_cluster['c2'] = []
        old_cluster['c3'] = []
        old_cluster['c4'] = []
        old_cluster['c5'] = []
        cluster = {}
        cluster['c1'] = []
        cluster['c2'] = []
        cluster['c3'] = []
        cluster['c4'] = []
        cluster['c5'] = []
        num = 1
        
        while True and num<max_iter:
            print("Clusetering Round {}".format(num))
            num+=1
            for key,value in vectors.items():
                self.uclidean_distance(centroids,value,key,cluster)
            
            if(old_cluster==cluster):
                break
            else:
                old_cluster = copy.deepcopy(cluster)
                centroids = []
                for key1,value1 in cluster.items():
                    print("Elements in cluster {} = {}".format(key1,len(value1)))
                    centre = self.calculate_centroid(value1,vectors)
                    centroids.append(centre)
                if(num<max_iter):
                    for key1,value1 in cluster.items():
                        value1.clear()
        
        text1 = ""
        for key,value in cluster.items():
            text1+="<div>{}</div>".format(key)
            text1+="<div>{}</div>".format(value)
        return text1
        answer = [0] * len(vectors)
        for key,value in cluster.items():
            for i in range(len(value)):
                ind = value[i].split('.')[1]
                try:
                    answer[int(ind)-1] = key
                except:
                    print(ind)
        for i in range(len(answer)):
            text+="{} ".format(answer[i])
            print(answer[i],end=' ')
            if(i%30 == 0):
                text+="\n"
                print("")
        return text



eel.init('web')
import time
@eel.expose                         # Expose this function to Javascript
def say_hello_py(text,func):
    knn = KNN()
    kmean = Kmeans()
    kmean = Kmeans()
    if func=='trainKNN':   
        eel.say_hello_js("Generating Vectors From 70% Data ...")
        knn.train_KNN()
        eel.say_hello_js("Vectors Generated Successfully")
        time.sleep(1)
        file = open('./KNN/testDocs.txt','r')
        test = json.load(file)
        file.close()
        text = "<h3><span style='color:yellow'>Seperated Test Files</span></h3><br/>"
        for key,value in test.items():
            text+="<h5 style='color:#F9008C'>{}</h5 style='color:#00A1F9'><p style='color:yellow'>{}</p>".format(key,value)
        return text
    elif func=='testKNN':
        eel.say_hello_js("Working on 30% Test Data ...")
        accuracy,confusion = knn.run_KNN()
        eel.say_hello_js("Testing Completed. Fetching Results")
        time.sleep(1) 
        text = "<div style='color:rgb(137, 202, 211);font-size:large;'>Accuracy Score = {}</div>".format(accuracy)
        text+="<div style='color:rgb(137, 202, 211);font-size:large;'>Atual on x, Predicted on y</div>"
        text+="<div style='color:rgb(137, 202, 211);'> At Cr Ft Rg Tn</div>"
        text+= "<div style='color:aliceblue;'>{}</div>".format(confusion)
        return text
    elif func=='predictKNN':
        eel.say_hello_js("Working On Your Document")
        time.sleep(1)
        pred = knn.predict_text(text)
        eel.say_hello_js("Here We Go...")
        text = "<div style='font-size: larger;font-weight: bolder; color:yellow'>Your Document Belongs To {} Class</div>".format(pred)
        return text 
    elif func=='trainKmeans':
        eel.say_hello_js1("Generating Vectors For Kmean Clustering")
        kmean.train_Kmean()
        eel.say_hello_js1("Kmeans Vectors Genrated")
        return
    elif func=='testKmeans':
        eel.say_hello_js1("Genrating Clusters Please Wait..")
        cluster = kmean.run_K_mean(max_iter=int(text))
        eel.say_hello_js1("Here We Go..")
        text = "<div style='color:yellow'>{}</div>".format(cluster)
        return text
    elif func == 'seeList':
        file = open('./KNN/testDocs.txt','r')
        data = json.load(file)
        text = "<h3><span style='color:yellow'>Seperated Test Files</span></h3><br/>"
        for key,value in data.items():
            text+="<h5 style='color:#F9008C'>{}</h5 style='color:#00A1F9'><p style='color:yellow'>{}</p>".format(key,value)
        return text
   # Call a Javascript function
eel.start('index.html', size=(1000, 1080))
