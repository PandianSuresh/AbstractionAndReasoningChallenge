# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:24:06 2020

@author: sures
"""

from semnets import SemNet
from visualize import load_tasks, plot_task
from glob import glob 
from matplotlib import pyplot as plt
from matplotlib import colors

task_files = glob('c:/OMSCS/kaggle/AbstractionAndReasoningChallenge/**/*.json')
tasks = load_tasks(task_files)
i = 3
(taskfile, task) = list(tasks.items())[i]
plot_task(list(tasks.items())[i])
print(len(task['train']))
SN = SemNet()
for k in range(len(task['train'])):
    train = task['train'][k]
    SN.train([train['input'],train['output']])
SN.plotsemnet()
test = task['test'][0]
# print(SN.predict(test['input']))

cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
i,th = 0,8
for frame in SN.frames:
    for node in frame.nodes:
        
        print(node.ndname)
        print('size',node.size)
        if node.relpos:
            print('relpos')
            for (a,b) in node.relpos:
                print(a,b.ndname)
        else:
            print('relpos',node.relpos)
        print('bbox',node.bbox)
        plt.imshow(node.rawnode, interpolation='nearest', cmap = cmap, norm = norm)
        plt.show()
        i += 1
        if i > th:
            break
    if i >th:
        break