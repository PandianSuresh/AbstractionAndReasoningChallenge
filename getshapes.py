# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:26:40 2020

@author: sures
"""
from semnets import SemNet, Frame, Node
import numpy as np

def getElements(inputOrOutput):
    img = np.array(inputOrOutput)
    potElements = np.unique(img)
    potElements = potElements[potElements!=0]
    print('potential elements', potElements)
    elements = []
    for element in potElements:
        temp = np.where((img == element), img, 0)
        elements.append(temp)
    return elements
    
    
    
train_input = [[2, 2, 2, 2, 2, 2, 2, 0, 0],
       [2, 0, 0, 0, 0, 0, 2, 0, 0],
       [2, 0, 0, 0, 0, 0, 2, 0, 0],
       [2, 0, 0, 2, 0, 0, 2, 0, 0],
       [2, 0, 0, 0, 0, 0, 2, 0, 0],
       [2, 0, 0, 0, 0, 0, 2, 0, 0],
       [2, 2, 2, 2, 2, 2, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0]]

train_output = [[2, 2, 2, 2, 2, 2, 2, 0, 0],
       [2, 4, 4, 4, 4, 4, 2, 0, 0],
       [2, 4, 4, 4, 4, 4, 2, 0, 0],
       [2, 4, 4, 2, 4, 4, 2, 0, 0],
       [2, 4, 4, 4, 4, 4, 2, 0, 0],
       [2, 4, 4, 4, 4, 4, 2, 0, 0],
       [2, 2, 2, 2, 2, 2, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0]]


elements_input = getElements(train_input)
print('Elements input', elements_input)

elements_output = getElements(train_output)
print('Elements output', elements_output)

sem = SemNet()
sem.addFrame('train input', train_input)
sem.addFrame('train output')
print([frame.fmname for frame in sem.frames])
# sem.getxfm()    