# -*- coding: utf-8 -*-
"""
	Script to turn a column of lists (in this case the categories column) into 
	sparse columns of binary values. Can be aggregated by date.

	This script was taken from the original Kaggle dataset:
	https://www.kaggle.com/hugodarwood/epirecipes?select=recipe.py
"""

#coding: utf8
import pandas as pd
import numpy as np

def sublist_uniques(data,sublist):
    categories = set()
    for d,t in data.iterrows():
        try:
            for j in t[sublist]:
                categories.add(j)
        except:
            pass
    return list(categories)

def sublists_to_binaries(data,sublist,index_key = None):
    categories = sublist_uniques(data,sublist)
    frame = pd.DataFrame(columns=categories)
    for d,i in data.iterrows():
        if type(i[sublist]) == list or np.array:
            try:
                if index_key != None:
                    key = i[index_key]
                    f =np.zeros(len(categories))
                    for j in i[sublist]:
                        f[categories.index(j)] = 1
                    if key in frame.index:
                        for j in i[sublist]:
                            frame.loc[key][j]+=1
                    else:
                        frame.loc[key]=f
                else:
                    f =np.zeros(len(categories))
                    for j in i[sublist]:
                        f[categories.index(j)] = 1
                    frame.loc[d]=f
            except:
                pass
                
    return frame