%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
import nltk
import re
import os
from six import iteritems
from nltk.stem.snowball import SnowballStemmer


def tokenize_and_stem(text, li):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    stems = [stemmer.stem(t) for t in filtered_tokens]
    stems1 = [x for x in stems if x not in li]
    return stems1


def tokenize_only(text, li):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_tokens1 = [x for x in filtered_tokens if x not in li]
    return filtered_tokens1



path = '/home/ignacio/Documents/chapter2/'
Bores = 'NamoiBoresWGS84.csv'
Lithologs = 'NGIS_LithologyLog.csv'

DF=pd.read_csv(path+Bores)
DF=DF.set_index('HydroCode')
DF2=pd.read_csv(path+Lithologs)
DF2=DF2.set_index('HydroCode')

DF2['Latitude']=DF.Latitude_W
DF2['longitude']=DF.Longitude_

Litho=DF2.dropna(subset=['Latitude', 'longitude'], how = 'all')
Litho = Litho[Litho.ToDepth < 7000]
new_data=Litho

## for Takuya (sample with 1000 rows)
sampled = new_data.sample(1000)
sampled.to_csv(path+'sampled_bores.csv')
new_data.to_csv(path+'boresTa.csv')


objectID=new_data.OBJECTID.tolist()
Descriptions=new_data.Description.tolist()
Descriptions = [str(n) for n in Descriptions]
remove = Descriptions.index('nan')
del Descriptions[remove]
del objectID[remove]

Dic=dict(zip(objectID, Descriptions))

# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer as variabled 'stemmer'
stemmer = SnowballStemmer("english")
# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed


stopw2=['redish', 'reddish', 'red', 'black', 'blackish', 'brown', 'brownish',
        'blue', 'blueish', 'orange', 'orangeish', 'gray', 'grey', 'grayish',
        'greyish', 'white', 'whiteish', 'purple', 'purpleish', 'yellow',
        'yellowish', 'green', 'greenish', 'light', 'very', 'pink','coarse',
        'fine', 'medium', 'hard', 'soft', 'coloured', 'multicoloured',
        'weathered', 'fractured']



totalvocab_stemmed = []
totalvocab_tokenized = []

for i in Descriptions:
    allwords_stemmed = tokenize_and_stem(i, stopw2) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend([allwords_stemmed]) #extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i, stopw2)
    totalvocab_tokenized.extend([allwords_tokenized])

print(np.shape(totalvocab_tokenized),  np.shape(totalvocab_stemmed))

## used gensim instead of cosdisimilarity from sklearn due to the huge distance matrix
dictionary = gensim.corpora.Dictionary(totalvocab_stemmed)
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(once_ids)
dictionary.compactify()
print(dictionary)
# dictionary.save(path+'/dictio.dict')
 # store the dictionary, for future reference

corpus = [dictionary.doc2bow(text) for text in totalvocab_stemmed]
print(corpus)
gensim.corpora.MmCorpus.serialize(path+'corpus.mm', corpus)  # store to disk, for later use
tf_idf = gensim.models.TfidfModel(corpus)
sims = gensim.similarities.Similarity(path, tf_idf[corpus],
                                      num_features=len(dictionary))
x, y =[],[]
for n, i in enumerate(corpus[0:20]):
    dist = 1-sims[tf_idf[i]]
    # print(dist, len(dist))
    if i == 0:
        x0, y0 = 0, 0
    elif i == 1:
        x0, y0 = 0, dist[0]
    else:
        dp1p2 = dist[0] + dist[1]
        dp1pn = dp1p2 + dist[1]
        dp2pn = dist[0] + dp1p2
        A = (dp1p2**2 + dp1pn**2 - dp2pn**2)/(2*dp1p2*dp1pn)
        x0, y0 = dp1pn*np.cos(A), dp1pn*np.sin(A)
    x.append(x0)
    y.append(y0)


plt.scatter(x, y)
for i,n in enumerate(Descriptions[0:20]):
    plt.text(x[i],y[i],n)
plt.show()

zip(x, y)
for n in zip(x,y):
    print(n)
for n in totalvocab_tokenized[:20]:
    print(n)


'''The problem seems that as we are converting to a 2D plane, and using the first
two descriptions as references ('sand' and 'clay'), in the plane different
categories are being plotted at the same location and distance from these descriptions.
Other issue is that it doesn't differentiate by the order in the description. Thus,
sandy clay is plotted at the same location that clayey sand...'''


Litho.columns
categories = Litho.MajorLithCode.unique()
print(len(categories), categories)

times = Litho.MajorLithCode.value_counts()>1
#removing categories that contains less than 6 descriptions
cleanLitho = Litho[Litho.groupby('MajorLithCode').MajorLithCode.transform(len) > 5]
cleanLitho.MajorLithCode.value_counts()
cats = cleanLitho.MajorLithCode.unique()

#checking the description for first category
cleanLitho[cleanLitho['MajorLithCode']==cats[0]][['MajorLithCode', 'Description']]



# if ('Basalt' in n) or ('basalt' in n):
#     clas == 'Basalt'
#
# elif ('Rhyolite' in n) or ('rhyolite' in n)\
# or ('Tuff' in n) or ('tuff' in n)\
# or ('Volcanic' in n) or ('volcanic' in n)\
# or ('Lava' in n) or ('lava' in n)\
# or ('Agglomerate' in n) or ('agglomerate' in n):
#     class = 'Extrusive'
#
# elif ('Granite' in n) or ('granite' in n) \
# or ('Diorite' in n) or ('diorite' in n):
#     class = 'Intrusive'
#
# elif ('Sandstone' in n) or ('sandstone' in n):
#     clas = 'Sandstone'
#
# elif ('Claystone' in n) or ('claystone' in n)\
# or ('Mudstone' in n) or ('mudstone' in n)\
# or ('Siltstone' in n) or ('siltstone' in n):
#     class = 'Mudrocks'
#
# elif ('Limestone' in n) or ('limestone' in n)\
# or ('Dolomite' in n) or ('dolomite' in n):
#     class = 'Limestones'
#
# elif ('Conglomerate' in n) or ('conglomerate' in n):
#     class == 'Conglomerate'
#
# elif ('Shale' in n) or ('shale' in n)\
# or ('Slate' in n) or ('slate' in n):
#     class == 'Shale'
#
#
#
# Litho[['MajorLithCode', 'Description']]
