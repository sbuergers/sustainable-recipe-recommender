# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:13:38 2020

@author: sbuer
"""

import os
import bioc


# Deserialize ``fp`` to a BioC collection object.
pathname = 'foodbase'
filename = 'FoodBase_uncurated'
suffix = '.xml'
with open(os.path.join(pathname, filename + suffix), 'r') as fp:
    collection = bioc.load(fp)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	