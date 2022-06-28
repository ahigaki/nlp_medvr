import numpy as np
import pandas as pd
from Bio import Medline
import matplotlib.pyplot as plt
from pathlib import Path
scr_path = Path(__file__).resolve().parent #get the abs path of this script

file = str(scr_path) + "/pubmed-virtualrea-set.txt"
with open(file) as recs:
    pubmeds = Medline.parse(recs)
    df = pd.DataFrame(pubmeds)
    abs_srs = df["AB"].dropna()
    #print(abs_srs[:10])

abs_ls = abs_srs.tolist()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#vect = CountVectorizer(min_df=5, ngram_range=(1,2))
vect = TfidfVectorizer(min_df=5, ngram_range=(1,2))
vect_abs = vect.fit_transform(abs_ls)
print("vect shape:", vect_abs.shape)

feat_names = vect.get_feature_names()
#print(feat_names[1000])

from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=2, random_state=0).fit_predict(vect_abs)

i=0
for doc, cls in zip(abs_ls, clusters):
    print("Cluster:",cls, doc[:200])
    i+=1
    if i==30:
        break