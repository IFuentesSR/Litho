%matplotlib inline
from scipy import stats, integrate
import seaborn as sns
import numpy as np
import pylab
import os
import pandas as pd
import matplotlib.pyplot as plt
import pyclust
from scipy.cluster.hierarchy import dendrogram, linkage
from kmodes import kmodes

path = os.getcwd()+'/'
DF = 'boresTa.csv'

DF=pd.read_csv(path+DF)
DF=DF.set_index('HydroCode')
new_data=DF
new_data.columns
new_data['Description'] = [str(n) for n in new_data.Description]



##trying to check any relation between MajorLithCode and depth
plot = sns.stripplot(x=new_data.MajorLithCode, y=new_data.ToDepth).set_yscale('log')





###Clustering by kmodes by using Despcriptions
array = new_data[['FromDepth', 'ToDepth', 'Latitude', 'longitude', 'Description']].as_matrix()
km = kmodes.KModes(n_clusters=20, init='Huang', n_init=10, verbose=1)
clusters = km.fit_predict(array)
# Print the cluster centroids
print(km.cluster_centroids_)
new_data['Kmodes'] = clusters
for n in range(0,19):
    new=new_data[new_data.Kmodes==n]
    print('\n\n\ncluster: '+str(n)+'\n------------------------\n', new.MajorLithCode.value_counts())
plot = sns.stripplot(x=new_data.Kmodes, y=new_data.ToDepth).set_yscale('log')





##Clustering by kmodes by using MajorLithCode
array2 = new_data[['FromDepth', 'ToDepth', 'Latitude', 'longitude', 'MajorLithCode']].as_matrix()
km2 = kmodes.KModes(n_clusters=20, init='Huang', n_init=10, verbose=1)
clusters2 = km2.fit_predict(array)
new_data['Kmodes2'] = clusters2
for n in range(0,19):
    new=new_data[new_data.Kmodes2==n]
    print('\n\n\ncluster: '+str(n)+'\n------------------------\n', new.MajorLithCode.value_counts())
plot = sns.stripplot(x=new_data.Kmodes2, y=new_data.ToDepth).set_yscale('log')




##kmedoids
# kmd = pyclust.KMedoids(n_clusters = 20)
# array.shape
#
# kmd.fit(array2)
#
# help(pyclust.KMedoids())



###Hierarchical Clustering
