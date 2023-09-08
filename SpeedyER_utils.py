def remove_empties(corp_in):
    # removes empty strings from an input corpus 

    outcorp = []
    # initialize a list for our reconstructed list of lists

    for i in range(len(corp_in)):
        row = corp_in[i]
        # pick off a row from the incoming corpus for each for loop iteration

        if '' in row:
            row = [e for e in row if e != '']
            # keep everything from the row except empty strings
        else:
            pass

        outcorp.append(row)
        # add the reformatted row to the building output corpus

    return outcorp

def embed_corpus(corp, model):
    # embeds an entire corpus using a pre-trained FastText model 
    for i in range(len(corp)):

        if i % 10000 == 0:
            print("We're on iteration ", i)

        pick = corp[i]
        emb = np.mean(model.wv[pick], axis = 0)
        emb = emb.reshape(1, len(emb))

        emb[pick == ''] = 0

        if i == 0:
            emb_corpus = emb
        else:
            emb_corpus = np.vstack([emb_corpus, emb])
    
    return emb_corpus

import math
from scipy.spatial import distance
import numpy as np

def topK_cosine(corp, K, model_choice):
    # this will allow us to find the top K matches based on cosine distance of embedded vector pairs
    # because this process evaluates the whole comparison pool, it can be run in parallel because each 
    # record has its distinct top K records candidate pairs
    for i in range(len(corp)):

        pick = corp[i]
        emb = np.mean(model_choice.wv[pick], axis = 0)
        emb = emb.reshape(1, len(emb))
        # here we get the embedded vector for our input record 
        # importantly, we aggregate by taking the vertical mean (e.g., [4, 150] --> [1, 150]) on each element

        if i == 0:
            emb_corpus = emb
        else:
            emb_corpus = np.vstack([emb_corpus, emb])

    all_pairs = 1 - distance.cdist(emb_corpus, emb_corpus, metric = "cosine")
    # this looks at all of the candidate pairs. 
    # It will be size n x m, where n is the number of incoming records, and m is the number of comparison records 
    top_k_pairs = np.argsort(-all_pairs)[:, :K]
    # this takes the top K scores from all_pairs and returns the indices where those scores live 
    # this will be n x K, where n is the number of incoming records, and K is the user-input number of nearest neighbors allowed 

    return all_pairs, top_k_pairs

import json
from faker import Faker
from nicknames import NickNamer
import random
import numpy as np
import math
import string
import pandas as pd
# libraries for generating information and then appropriate nicknames, respectively.

def gen_people(num_people, cut_frac, numd, stoch_frac, restart = 0):
    # Here we will generate records using number of people and stochasticity to influence messiness and size of set
    # restart is used so we can call genpeople multiple times and keep unique entity ID numbers
    
    nn = NickNamer()
    fake = Faker()
    # initializing generators for easier calls later on

    f = open('states.json')
    states = json.load(f)
    reverse_states = {v: k for k, v in states.items()}
    # grab the json with state abbreviations and read it, reversing abbreviations for full names

    reverse_states

    rec_num = -1
    # initialize record number at 0
    
    for p in range(num_people):
        
        rec_num += 1

        if p % 1000 == 0:

            print("Sample %d of %d total" % (p, num_people))

        person = np.array([fake.first_name(), fake.first_name(),fake.last_name(), ''.join(random.choices(string.digits, k = 9)), fake.state(), ''.join(random.choices(string.digits, k = 10)), fake.state(), p + restart])
        # here we are picking a fake first and last name, fake middle initial, fake ssn, fake birth and license states, and a fake driver's license number

        if p == 0:
            people = person
            # the first person will initialize an array structure for us
        else:
            people = np.vstack([people, person])
            # otherwise, we're just adding rows on rows 

        numdupes_p = random.randint(0, numd + 1)
        # we can take a user parameter to pick a random integer number of duplicates per record

        for q in range(numdupes_p):
            # now we'll iterate over the number of duplicates for a given record on hand\
            
            rec_num += 1

            new_person = person.copy()
            #new_person[8] = rec_num
            change_entries = random.uniform(0, stoch_frac)
            changes = random.sample(range(new_person.shape[0] - 1), math.floor(change_entries*(new_person.shape[0])))
            # here we're employing the same strategy as above, but now we are shifting some random fraction of field records
            # i leave out the userID on purpose here

            for r in range(len(changes)):
                
                if changes[r] == 0:
                    # first name case, will make more robust when we have keys built in
                    
                    nicks = list(nn.nicknames_of(new_person[0]))
                    
                    choice = random.randint(0, 10)
                    
                    if choice == 3 or choice == 5:
                        
                        new_person[0] = ''.join(random.sample(new_person[0],len(new_person[0])))
                    
                    elif choice == 1 or choice == 4 or choice == 2:
                        # an alternative to make this a little harder for the algorithm is to misspell things deliberately
                        nm = list(new_person[0])

                        if len(nm) < 3:
                            nm = nm[0]
                            new_person[0] = nm
                        else:
                            nm = "".join(nm[:-2])
                            new_person[0] = nm        
                    
                    else: 
                        
                        if len(nicks) > 0:
                            new_nickname = random.choice(nicks)
                            # we can only act on nicknames that exist :)
                            new_person[0] = new_nickname.capitalize()
                        
                        else:
                            pass

                    
                elif changes[r] == 1:
                    # last name case, will make more robust when we have keys built in
                    
                    choice = random.randint(0, 5)
                    
                    if choice == 5:
                        new_person[1] = fake.first_name()
                        # this shouldn't be all that overwhelmingly likely
                    else:
                        pass

                    
                elif changes[r] == 2:
                    # middle name case
                    
                    choice = random.randint(0, 10)
                    
                    if choice in [1, 2, 3, 4, 5]:

                        new_person[2] == fake.last_name()

                    elif choice == 6 or choice == 7:

                        new_person[2] = ''.join(random.sample(new_person[2],len(new_person[2])))   

                    else:
                        pass
                        
                        
                elif changes[r] == 3:
                    # SSN case
                    
                    replace_ssn = random.uniform(0, 0.35)
                    ssn_changes = random.sample(range(len(new_person[3])), math.floor(replace_ssn*(len(new_person[3]))))
                    new_ssn = list(new_person[3])
                    # convert to list for easy indexing and assignment
                    
                    choice = random.randint(0, 3)
                    
                    if choice in [0, 1, 2, 3]:
                        
                        for ch in range(len(ssn_changes)):

                            new_ssn[ssn_changes[ch]] = str(random.randint(0, 9))

                        new_person[3] = "".join(new_ssn)
                        # convert back to a string for storage
                        
                
                elif changes[r] == 4:
                    # DL state case
                    
                    choice = random.randint(0, 5)
                    
                    if choice == 5:
                        newchoice = random.randint(0, 1)
                        if newchoice == 0:
                            new_person[4] = reverse_states[new_person[4]]
                        else: 
                            pass
                    else:
                        newchoice = random.randint(0, 1)
                        newstate = fake.state()
                    
                        if newchoice == 0: 
                            new_person[4] = newstate
                        else:
                            new_person[4] = reverse_states[newstate]                 
                
                elif changes[r] == 5:
                    # DL num case
                    
                    replace_dl = random.uniform(0, 0.35)
                    dl_changes = random.sample(range(len(new_person[5])), math.floor(replace_dl*(len(new_person[5]))))
                    new_dl = list(new_person[5])
                    # convert to list for easy indexing and assignment
                    
                    for ch in range(len(dl_changes)):
                        
                        new_dl[dl_changes[ch]] = str(random.randint(0, 9))
                        
                    new_person[5] = "".join(new_dl)
                    # convert back to a string for storage
                    
                elif changes[r] == 6:
                    # Birth state case
                    
                    choice = random.randint(0, 1)
                    
                    if choice == 0:
                        # birth state should be pretty stable
                        pass
                    elif choice == 1:
                        new_person[6] = reverse_states[new_person[6]]
                    
            people = np.vstack([people, new_person])     
            # add in our new synthetic record 
                
    for f in range(len(people)):

        final_person = people[f]
        cut_entries = random.uniform(0, cut_frac)
        edit = random.sample(range(final_person.shape[0] - 2), math.floor(cut_entries*(final_person.shape[0])))
        people[f][edit] = ''
        # we'll cut out some records for additional variety. This is tunable with the cut_fraction parameter

    people_df = pd.DataFrame(people, columns = ['firstName', 'middleName', 'lastName', 'ssn', 'dlState', 'dlNumber', 'birthState', 'trueID'])
    
    pd.DataFrame.iteritems = pd.DataFrame.items
    
    for (columnName, columnData) in people_df.iteritems():
        people_df[columnName] = people_df[columnName].str.lower()
        # sending to lower case has been shown to make models more generalizable because there is less meaning to capital letters 
        # and text can get jumbled, but still understood 

    return people_df

