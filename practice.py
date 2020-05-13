import json
import math
file = open('./Kmean/vector.json')
data = json.load(file)
file.close()

for key,value in data.items():
    print(len(value))
    
exit()
# file = open('./Kmean/Corpus.txt')
# corpus = file.read().split(',')
# file.close()
# file = open('./Kmean/DocumentFrequency.txt')
# df = file.read().split(',')
# file.close()

# file = open('./Kmean/all_doc.json')
# all_doc = json.load(file)
# file.close()
# file = open('./Kmean/all_tf.json')
# all_tf = json.load(file)
# file.close()
# for key, value in all_doc.items():
#     for i in range(len(value)):
#         tf = all_tf[key][i]
#         #print("Termfrequency {}".format(all_tf[key][i]))
#         for j in range(len(corpus)):
#             if corpus[j]==value[i]:
#                 print(corpus[j])
#                 docf = df[j]
#                 #print("DocumentFreq {}".format(docf))
#                 print(int(tf)* math.log10(  len(all_tf)/int(docf) ))
#                 print(data[key][j])
#     exit()            




import numpy as np
M = np.array(data[list(data.keys())[0]])
count = 0
clss = [] 
for key,value in data.items():
    clss.append(key)
    if(count != 0):
        M = np.vstack((M,value))
    count = 1

from sklearn.cluster import KMeans
kmean = KMeans(n_clusters=5,random_state=0).fit(M)
lab = (kmean.labels_)
ans = (zip(lab,clss))
print(sorted(ans, key=lambda x: x[0]))


# for l in lab:
#     print(l)

# file = open('./Kmean/Vector.csv','w')
# file1 = open('./Kmean/Corpus.txt')
# file.write('label,')
# corp = file1.read().split(',')
# for i in range(len(corp)):
#     file.write("{},".format(corp[i]))
# file.write('\n')
# for key,value in data.items():
#     lab = key.split('.')[0]
#     file.write("{},".format(lab))
#     for i in range(len(value)):
#         file.write("{},".format(value[i]))
#     file.write('\n')

# import pandas as pd

# data = pd.read_csv('./Kmean/Vector.csv')
# print(data.head())


# file = open('./KNN/Corpus.txt')
# data = file.read().split(',')
# print(len(data))