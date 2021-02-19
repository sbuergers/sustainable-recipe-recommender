# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:51:56 2020

@author: sbuer
"""

units_ml = {'drop':0.051,
				'smidgen':0.116,
				'pinch':0.231,
				'dash':0.462,
				'saltspoon':0.924,
				'coffeespoon':1.848,
				'fluid dram':3.697,
				'teaspoon':4.93,
				'dessertspoon':9.86,
				'tablespoon':14.79,
				'ounce':29.57,
				'wineglass':59.15,
				'teacup':118.29,
				'cup':236.59,
				'pint':473.18,
				'quart':946.35,
				'pottle':1892.71,
				'gallon':3785.41,
				'piece':118.29, # this is extremely variable (e.g. ginger, fish, pig, ...)
				'clove':4.93, # A clove of garlic should be around 1 tsp
				'envelope':1.25*29.57, # for yeast
				'pound':16*29.57, # 16 ounces
				'bunch':1.5*29.57, # 1-2 ounces
				'gram':1, # approximately correct, depending on material
				'package':19*29.57, # package of tofu... there is great variability though
				'head':500, # at least a pound for cabbage, cauliflower etc.
				'slice':0.8*29.57, # for a slice of cheese
				'sprig':1.848, 
				'can':12*29.57, # for a small can
				'stick':4*29.57, # for a butter stick
				'strip':29.57, 
				'stalk':59.15, # for a stalk of celery (could also be lemongrass)
				'cube':4.93, # should be 1 teaspoon (always sugar)
				'fillet':100, # 100 grams is roughly 1 fillet
				'handful':118.29, # by definition it's half a cup
				'fistfull':59.15, # by definition it's half a handful
				'bag':11*29.57, # 10-12 ounces it seems, though probably not always
				'loaf':6*236.59, # approximately 4-8 cups
				'bulb':8*29.57, # for a bulb of fennel
				'bottle':1.25*473.18, # could be beer or wine
				'ear':(3/4)*236.59, # for an ear of corn
				'ball':236.59, # a ball of mozzarella is a cup
				'batch':473.18, # pretty unclear, milk, eggs, fish, fruit...
				'sheet':14*29.57, # sheet of pastry dough or cheese
				'dozen':473.18, # used for clams - roughly one pound
				'liter':1000,
				'box':473.18, # e.g. box of chocolate or milk
				'packet':59.15, # can be many sizes, usually quite small (1/4 ounce), but can also be a packet of dumpling wrappers
				'chunk':1.1*14.79, # used for ginger pretty much
				'rack':3.5*16*29.57, # a rack of ribs... 3-4 pounds
				'jar':236.59, # pretty variable...
				'stem':14.79, # a stem of thyme
				'part':118.29, # bread, fruit and sugar... no clear amount, few entries though
				'branch':14.79,
				'inch':1.1*14.79, # used only for ginger
				'wedge':6*29.57, # a wedge of cheese is 4-8 ounces
				'link':29.57, # a link of sausage is 1 ounce
				'square':29.57, # a square of chocolate - 1 ounce
				'knob':118.29, # same as piece
				'scoop':59.15, # as much as a wineglass by definition
				'fifth':757, # a fifth of a gallon for liquor - wow
				'twist':29.57, # refers to lemon or orange peel
				'pair':172*2 # because in this case it's chicken breast !!!
				}