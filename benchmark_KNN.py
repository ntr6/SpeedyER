import numpy as np
import pandas as pd
from SpeedyER_utils import gen_people, remove_empties, topK_cosine
from gensim.models import Word2Vec, Doc2Vec, FastText
import time

fullkp2 = gen_people(1500, 0.3, 4, 0.85)

newdat = fullkp2.drop('trueID', axis = 1)

newcorp = newdat.values.tolist()

prunedcorp = remove_empties(newcorp)

ksize = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

recall_vec = []
clust_frac = []

for pp in range(len(ksize)):

    print("K fraction =", ksize[pp])

    all_p, top_k = topK_cosine(newcorp, int(len(newcorp)*ksize[pp]), model_choice = model)

    truekp = 0
    clusterkp = 0

    clust_frac.append(ksize[pp])

    for qq in range(len(top_k)):

        pickset = fullkp2.iloc[top_k[qq]]

        pickrow = fullkp2.iloc[qq]

        clustered_count = sum(pickset.trueID == pickrow.trueID)
        true_count = sum(fullkp2.trueID == pickrow.trueID)

        truekp += true_count
        clusterkp += clustered_count

    recall_vec.append(clusterkp/truekp)

    print(clusterkp/truekp)

print(recall_vec)