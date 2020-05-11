# import json
# file = open('./Kmean/vector.json')
# data = json.load(file)
# file.close()
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


file = open('./KNN/Corpus.txt')
data = file.read().split(',')
print(len(data))