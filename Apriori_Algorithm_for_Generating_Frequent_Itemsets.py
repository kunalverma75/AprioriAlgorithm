#!/usr/bin/env python
# coding: utf-8

# # Apriori Algorithm for Generating Frequent Itemsets

# ##### by Kunal Verma 

# ## Introduction

# Apriori invented by Rakesh Agarwal and Ramakrishnan Srikant in 1994 is
# a well known algorithm in data mining. Apriori Algorithm is used in finding frequent itemsets. Identifying associations between items in a dataset of transactions can be useful in various data mining tasks. The challenge is that given a dataset D having T transactions each with n number of attributes, how to find itemsets that appear frequently in D?
# 
# 
# Using this algorithm we are trying to find significant terms obtained from a collection of electronic document which can be further used for text classification, topic finding and many other applications.

# ## Definitions

# 
# 1) Let T = {$t_i, t2,..., t_N $} be the set of all transactions and I = {$h, h i_d$} be the set of all items in a transaction database. Each transaction $t_j$ consists of items which are subsets of I. 
# 
# 
# 2) Itemset: It is defined as a collection of zero or more items in a transaction. If an itemset has no items in it then it is termed as a null itemset, and if it contains k items then it is referred as a k-itemset. 
# 
# 3) Support count: Support count is defined as the number of transactions that contain a particular itemset. It is the most important property of an itemset. 
# 
# 

# Now using the rules of association and concepts of support and confidence, we are going to describe how the Apriori actually works.
# 

# ## Apriori Algorithm

# Apriori works on the following principle which is also known as apriori property:
# 
# If an itemset is frequent, then all of its subsets must also be frequent.
# 
# The Apriori algorithm needs a minimum support level as an input and a data set. The algorithm will generate a list of all candidate itemsets with one item. The transaction data set will then be scanned to see which sets meet the minimum support level. The sets which are below the specified minimum support level are removed from further computations.
# Now we will demonstrate the step-by-step approach of the algorithm using an example.

# | TID | Items   |
# |-----|---------|
# | 100 | 1,3,4   |
# | 200 | 1,3,4   |
# | 300 | 1,2,3,5 |
# | 400 | 2,5     |

# Our dataset contains four transactions which could be sentences in case of documents with above particulars (sets/words in that sentence)

# #### Step 1: Generating 1-itemset table

# Generate the 1-itemset table by enumerating all the unique elements of items. Correspondingly calculate their support 

# | Itemset | Support |
# |---------|---------|
# | {1}     | 2       |
# | {2}     | 3       |
# | {3}     | 3       |
# | {4}     | 1       |
# | {5}     | 3       |

# Support value is calculated as support ($a -> b$) = $\frac{\text{Number of transactions a and b appear}} {\text{total transactions}}$. But for the sake of simplicity we use support value as number of times each transaction appears. We also assume support threshold = 2.

# Now as we can see the support of Itemset 4 is 1 which is less then our threshold of 2. So we remove that itemset.
# Now we create the 2-itemset table

# #### Step 2 : Generate 2-itemsets table

# From the above 1-itemset table, we look at the different combinations of size 2 itemsets, not considering the eliminated itemsets. Correspondingly calculating the support of each 2-itemset

# | Itemset | Support |
# |---------|---------|
# | {1,2}   | 1       |
# | {1,3}   | 2       |
# | {1,5}   | 1       |
# | {2,3}   | 2       |
# | {2,5}   | 3       |
# | {3,5}   | 2       |

# Now repeat the above process of removing itemsets with support less than 2. and using the new 2-itemset create 3-itemset table. In this example {1,2} and {1,5} are removed.

# #### Step 3 : Generate 3-itemsets table

# The corresponding 3-itemseet table would be formed using all combinations of above accepted size 2 itemsets

# | Itemset | Support |
# |---------|---------|
# | {2,3,5} | 2       |

# Now we have to stop because 4-itemsets cannot be generated as there are only three items left.
# 
# Following are the frequent itemsets that we have generated and which are above support threshold: {1}, {2}, {3}, {5}, {1, 3}, {2, 3}, {2, 5}, {3, 5} and {2, 3, 5}

# ### Pseudo-code for the whole Apriori algorithm

# ![alt text](Steps.png "Apriori Example")

# 1) Create a list of candidate itemsets of length k
# 
# 2) Scan the dataset to see if each itemset is frequent
# 
# 3) Keep frequent itemsets to create itemsets of length k+1

# ## Association analysis

# Looking for hidden relationships in large datasets is known as association analysis or association rule learning. The problem is, finding different combinations of items can be a time-consuming task and prohibitively expensive in terms of computing power.
# 
#  Association rules suggest that a strong relationship exists between two items.
# The support and confidence are ways we can quantify the success of our association analysis.
# 
# The support of an itemset is defined as the percentage of the dataset that contains this itemset.
# 
# 
# The confidence for a rule P ➞ H is defined as support(P | H)/ support(P). Remember, in Python, the | symbol is the set union; the mathematical symbol is U. P | H means all the items in set P or in set H.

# 1) To find association rules, we first start with a frequent itemset. We know this set of items is unique, but we want to see if there is anything else we can get out of these items. One item or one set of items can imply another item.

# 2) We use this to find association between the above generated frequent itemsets. Also we calculate confidence for each association and prune those with confidence below the minimum threshold

# This gives you three rules: {1} ➞ {3},{5} ➞ {2},and {2} ➞ {5}. It’s interesting to see that the rule with 2 and 5 can be flipped around but not the rule with 1 and 3

# ## Code for the Apriori Algorithm and Association Rules

# ## Apriori algorithm

# ![alt text](apriori.png "Apriori Example")

# The following code is referenced from the blog on Apriori Algorithm on adataanlyst.com 
# Scanning the dataset
# For each transaction in the dataset:
# 
# For each candidate itemset, can:
# 
#     Check to see if can is a subset of tran
# 
#     If so increment the count of can
# 
# For each candidate itemset:
# 
# If the support meets the minimum, keep this item
# 
# Return list of frequent itemsets

# In[20]:


from IPython.display import Latex


# In[10]:


from numpy import *


# Dataset for testing

# In[13]:


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# This creates 1-itemset table(C1). we’ll scan the dataset to see if these one itemsets meet our minimum support requirements. The itemsets that do meet our minimum requirements become L1. L1 then gets combined to become C2 and C2 will get filtered to become L2.
# 
# Frozensets are sets that are frozen, which means they’re immutable; you can’t change them. You need to use the type frozenset instead of set because you’ll later use these sets as the key in a dictionary.

# In[8]:


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return list(map(frozenset, C1))


# Next function takes three arguments: a dataset, Ck, a list of candidate sets, and minSupport, which is the minimum support you’re interested in. This is the function you’ll use to generate L1 from C1.

# In[11]:


def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData


# In[14]:


dataSet = loadDataSet()
dataSet


# In[15]:


C1 = createC1(dataSet)
C1


# In[17]:


#D is a dataset in the setform.

D = list(map(set,dataSet))
D


# Now that you have everything in set form, you can remove items that don’t meet our minimum support.

# In[18]:


L1,suppDat0 = scanD(D,C1,0.5)
L1


# Creating candidate itemsets: Ck
# 
# The function aprioriGen() will take a list of frequent itemsets, Lk, and the size of the itemsets, k, to produce Ck

# In[19]:


def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList


# it will take the itemsets {0}, {1}, {2} and so on and produce {0,1} {0,2}, and {1,2}

# In[26]:


def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


# Now we run this main function apriori for implementing the apriori algorithm.

# ## Mining association rules

# The generateRules() function takes three inputs: a list of frequent itemsets, a dictionary of support data for those itemsets, and a minimum confidence threshold.

# In[21]:


def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


# calcConf() calculates the confidence of the rule and then find out the which rules meet the minimum confidence.

# In[24]:


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


# rulesFromConseq() generates more association rules from our initial dataset. This takes a frequent itemset and H, which is a list of items that could be on the right-hand side of a rule.

# In[27]:


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
L,suppData= apriori(dataSet,minSupport=0.5)
rules= generateRules(L,suppData, minConf=0.7)


# ## Limitations

# 1) Apriori algorithms can be very slow on very large datasets
# 
# 2) If the transaction datababse has 10,000 frequent 1-itemsets, they will generate $10^7$ candidate 2-itemsets even after employing the downward closure.
# 
# 3) Computing support and comparing with minimum support, the database needs to scanned at every level.

# ## Methods for Improving

# 1) Hash-based itemset counting
# 
# 2) Transaction reduction
# 
# 3) Partitioning
# 
# 4) Sampling
# 
# 5) Dynamic itemset counting

# ## Advantages/Disadvantages

# Pros:
# 
# 1) Uses large itemset property
# 
# 2) Easilly parallelized
# 
# 3) Easily implementable
# 
# Cons:
# 
# 1) Assumes transaction database is memory resident.
# 
# 2) Requires many database scans
# 
# 

# In[ ]:




