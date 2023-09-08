import numpy as np
import pandas as pd
from SpeedyER_utils import gen_people, remove_empties
from gensim.models import Word2Vec, Doc2Vec, FastText
import time

for i in range(90):

    if i == 0:
        restart = 0
    else:
        pass

    if i % 10 == 0:
       print("on iteration", i)
    else:
        pass

    df = gen_people(500, 0.3, 10, 0.85, restart)
    # we'll make a new dataframe following the described parameters

    restart = int(max(df.trueID)) + 1
    # we'll restart our indices for true cluster IDs with each loop

    if i == 0:
        dfkp = df
    else:
        dfkp = pd.concat([dfkp, df], ignore_index = True)
        print(len(dfkp))


corpus = dfkp.values.tolist()
# this will make a list of lists 

pruned_corpus = remove_empties(corpus)

tic = time.perf_counter()
original_model = FastText(vector_size = 400, window = 1, min_count = 1, sentences = corpus, epochs = 30)
toc = time.perf_counter()

print("Model training time for original model in seconds was", toc - tic)

tic = time.perf_counter()
pruned_model = FastText(vector_size = 400, window = 1, min_count = 1, sentences = pruned_corpus, epochs = 30)
toc = time.perf_counter()

print("Model training time for pruned model in seconds was", toc - tic)

