# -*- coding: utf-8 -*-
"""
Created on Sat May  2 07:55:58 2020

@author: sures
"""

import numpy as np
# import copy

class Node:
    """A 'Node' in a given 'Frame' of a 'Semantic Network'. 
       rawnode preserves the image as is
       shape stores the positional values but not the colour
       trmshape stores the bezel-less, colorless version of the image with all all-zero rows and cols removed
       clrtrmshape stores the bezel-less, colored version of the image with all all-zero rows and cols removed"""
    def __init__(self, node, ndname=0):
        self.ndname = ndname
        self.rawnode = node
        self.shape = np.where(node == 0, node, 1)
        color = np.unique(node)
        [self.color] = color[color != 0]
        self.xfm = ('added',0)
        self.relpos = []
        self.trmshape = self.trim(self.shape)
        self.clrtrmshape = self.trim(self.rawnode)
        self.size = self.trmshape.shape[0]*self.trmshape.shape[1]
        self.bbox = self.bbox2() 
        
    
    def trim(self,img):
        """Trims the margins leaving the bezel-less shape behind"""
        trm = img[~np.all(img == 0, axis=1)]
        trm = trm[:, ~np.all(trm == 0, axis=0)]
        return trm
    
    def bbox2(self):
        rows = np.any(self.shape, axis=1)
        cols = np.any(self.shape, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
    
        return [[rmin, cmin], [rmax, cmax]]
            
        
    
class Frame:
    """ A 'Frame' in a 'Semantic Network'. A 'Frame' is a collection of nodes and their relationships
    in that 'Frame'. """
    def __init__(self, fmname,numTotNodes,rawFrame):
        self.fmname = fmname
        self.rawFrame = rawFrame
        self.nodes = []
        self.segmentFrame()
        self.nodes.sort(key=lambda x:x.size, reverse=True)           
        i = numTotNodes
        for node in self.nodes:
            node.ndname = i
            i+=1
        # for i in range(1,len(self.nodes)):
        #     temp = self.nodes[:i]
        #     for anode in reversed(temp):
        #         anode.relpos = [(self.getrelpos(anode, refnode),refnode.ndname) for refnode in temp]
                
        for i in range(len(self.nodes),1,-1):
            anode = self.nodes[i-1]
            for j in range(i-1):
                refnode = self.nodes[j]
                anode.relpos.append((self.getrelpos(anode, refnode),refnode))
        
        
    def segmentFrame(self):
        potElements = np.unique(self.rawFrame)
        potElements = potElements[potElements!=0]
        
        potNodescollns = []   #each element is Potentially a collection of nodes
        for element in potElements:
            potNodescollns.append(np.where((self.rawFrame == element), self.rawFrame, 0))
        #To Do:: Segment disjointed cells of same colour into different nodes.
        for potNodescolln in potNodescollns:
            nodes = self.spatialSegment(potNodescolln)     #return spatially segmented symbols even if colors are the same       
            for node in nodes:
                anode = Node(node)
                # if len(self.nodes) != 0:
                    # anode.relpos = [(refnode.ndname,self.getrelpos(anode, refnode)) for refnode in self.nodes]  #getrelpos returns a list of relative positions to other already identified nodes.
                self.nodes.append(anode)
                

    def spatialSegment(self,myimg):
        """This method implements a sort-of watershed based segmentation of same coloured spatially
        seperated shapes. inspired by https://www.cmm.mines-paristech.fr/~beucher/wtshed.html. This
        approach initializes every coloured cell to a number from a sequence. The uncoloured cells are
        left with 0. every neighbour of a given coloured cell is set to the same colour as itself,
        ignoring the uncoloured cell. in the end, all spatially contiguous cells are left with the same
        colour but different from other spatially contiguous cells.
        
        arguments: 
            myimg: a single coloured image for spatial segmentation
        
        return:
            nodes: a list of images containing one symbol(potentially) each """
    # def spatialSegment(myimg):
        val = (np.unique(myimg)[1])
        # print(val)
        b = np.where(myimg == val)
        coord = list(zip(b[0],b[1]))
        temp = np.copy(myimg)
        i = 1
        for el in coord:
            (x,y) = el
            temp[x][y]=i
            i += 1
        # print(temp)
        
        explored = []
        img = np.copy(temp)
        frontier=[]
        cnt = 0    
        while True:
            if len(frontier) == 0:
                frontier.append(coord.pop())
            neighbor = []
            cnt += 1        
            # print('curr coord', a)
            a = frontier.pop()
            (r,c) = a
            curval = img[r][c]
            up,down,right,left = False,False,False,False
            if r>0: #up
                up = True
                x = r-1
                y = c
                if img[x][y] != 0 and img[x][y] != curval: neighbor.append((x,y))
            if r<len(img)-1: #down
                down = True
                x = r+1
                y = c
                if img[x][y] != 0 and img[x][y] != curval: neighbor.append((x,y))
            if c>0: #left
                left = True
                x = r
                y = c-1
                if img[x][y] != 0 and img[x][y] != curval: neighbor.append((x,y))
            if c<len(img[0])-1: #right
                right = True
                x = r
                y = c+1
                if img[x][y] != 0 and img[x][y] != curval: neighbor.append((x,y))
            if up and right:
                x,y = r-1,c+1
                if img[x][y] != 0 and img[x][y] != curval: neighbor.append((x,y))
            if up and left:
                x,y = r-1,c-1
                if img[x][y] != 0 and img[x][y] != curval: neighbor.append((x,y))
            if down and right:
                x,y = r+1,c+1
                if img[x][y] != 0 and img[x][y] != curval: neighbor.append((x,y))
            if down and left:
                x,y = r+1,c-1
                if img[x][y] != 0 and img[x][y] != curval: neighbor.append((x,y))
                
            # print('neighbors',neighbor)
                   
            for neigh in neighbor:
                (nr,nc) = neigh
                img[nr][nc] = img[r][c]
                coord.remove(neigh)
                frontier.append(neigh)
            # print('frontier',frontier)
            explored.append(a)
            # print('explored',explored)
            if len(coord) == 0: 
                # print('count',cnt)
                break
        # print(img)
        # print(coord)
        
        potElements = np.unique(img)
        potElements = potElements[potElements!=0]
        nodes = []
        for element in potElements:
            nodes.append(np.where((img == element), myimg, 0))
        
        # i = 1
        # for node in nodes:
        #     print('node', i,':')
        #     print(node)
        #     i +=1
        return nodes
        
    def getrelpos(self, anode, refnode):
        """ attempts to identify the relative position of a given node 'anode' with respect to a 
        reference 'refnode'. If not successful, returns []"""
        #node.bbox returns [[rmin, cmin], [rmax, cmax]]
        relpos = []
        if (anode.bbox[0][1] <= refnode.bbox[1][1] and refnode.bbox[0][1] <= anode.bbox[1][1]
            and anode.bbox[0][0] <= refnode.bbox[1][0] and refnode.bbox[0][0] <= anode.bbox[1][0]):   #anode 
            relpos.append('isOverlapping')    
        if np.all(anode.bbox[0] > refnode.bbox[0]) and np.all(anode.bbox[1] < refnode.bbox[1]):
            relpos.append('isInside')
        if anode.bbox[1][0] <= refnode.bbox[0][0]:   #anode rmax <= refnode rmin
            relpos.append('isAbove')
        if anode.bbox[0][0] >= refnode.bbox[1][0]:   #anode rmin >= refnode rmax
            relpos.append('isBelow')
        if anode.bbox[1][1] <= refnode.bbox[0][1]:   #anode cmax <= refnode cmin   
            relpos.append('isLeftof')
        if anode.bbox[0][1] >= refnode.bbox[1][1]:   #anode cmin >= refnode cmax
            relpos.append('isRightof')
        
            
        #ToDo - identify relative positions in terms of
        #'inside', 'left of', 'right of', 'above', 'below'
        #'left of & above', 'left of & below'
        #'right of & above', 'right of & below'
        
        return relpos
            
class SemNet:
    """A 'Semantic Network' representation of collection of elements ('Nodes') and
    their transformations across 'Frames'. """
    def __init__(self):
        self.frames = []
        self.numFrames = 0
        self.numTotNodes = 0
        
    def train(self,trainPair=[]):
        
        framename = ['train input','train output']
        for rawFrame, name in zip(trainPair, framename):
            aframe = Frame(name+str(self.numFrames),self.numTotNodes,rawFrame)
            aframe = self.fixnodenames(aframe)
            self.frames.append(aframe)
            self.numFrames += 1
            self.numTotNodes += len(aframe.nodes)
            
        self.getxfm()
        # self.plotsemnet()
    def fixnodenames(self, aframe):
        """ This module comapres the new symbols with existing nodes with the objective of finding nodees
        with similar properties by imposing certain "similarity constraints". The colour-neutral trimmed
        shape of the symbols is an example of a similarity constraint. Symbols fulfiling similarity 
        constraints are given a common name"""
        for anode in aframe.nodes:
            for refframe in self.frames:
                for refnode in refframe.nodes:
                    if np.all(anode.clrtrmshape == refnode.clrtrmshape):
                        anode.ndname = refnode.ndname
        return aframe
    
    def getxfm(self):
        """To be called after a SemNet is supplied with a test or train input/output frame pair.
        This method goes through every node in the input frame matching it with nodes in output frame.
        A match is detected when a node in the input frame is applied a transformation rule resulting
        in a match with a node in the output frame. When a match is identified, the names of the nodes
        are set to a common name to indicate they are the same element.""" 
        for anode in self.frames[-2].nodes:
            # match = False
            anode.xfm = ('deleted',0)
            for refnode in self.frames[-1].nodes:
                #ToDo: Add more transformation detectors
                # print('anode.shape',anode.shape,'refnode.shape',refnode.shape)
                if (anode.shape == refnode.shape).all():
                    refnode.ndname = anode.ndname
                    # match = True
                    if anode.color == refnode.color:
                        anode.xfm = refnode.xfm = ('unchanged',0)
                    else:
                        anode.xfm = refnode.xfm = ('color',refnode.color)
                if np.all(anode.clrtrmshape == refnode.clrtrmshape):
                    refnode.ndname = anode.ndname
                    anode.xfm = refnode.xfm = ('unchanged',0)
        
            
                
                
    def predict(self,rawFrame):
        
        #To Do::Finish up prediction code
        aframe = Frame('test input',self.numTotNodes,rawFrame)
        aframe = self.fixnodenames(aframe)
        self.frames.append(aframe)
        self.numFrames += 1
        self.numTotNodes += len(aframe.nodes)
        return rawFrame
    
    def plotsemnet(self):
        """ Produces a graphical representation of semantic network.
        In this implementation, graphical represenation only considers two frames per semantic network"""
        a = iter(self.frames) #iter and zip is needed to extract input and output as pairs in every iteration
        # a = self.frames
        edgelist = []
        edgelabels = []
        for aframe,refframe in zip(a,a):
            anodes = aframe.nodes
            refnodes = refframe.nodes
            
            for anode in anodes:
                edgelist.append((aframe.fmname+':'+str(anode.ndname),
                                 refframe.fmname+':'+str(anode.ndname)))
                edgelabels.append(anode.xfm)
            for refnode in refnodes:
                if refnode.xfm == ('added', 0):
                    edgelist.append((aframe.fmname+':'+str(refnode.ndname),
                                     refframe.fmname+':'+str(refnode.ndname)))
                    edgelabels.append(refnode.xfm)
            
        for i in range(len(edgelist)):
            print(edgelist[i][0],'-----',edgelabels[i],'---->',edgelist[i][1]) 

        
                
                
# train_input = [[2, 2, 2, 2, 2, 2, 2, 0, 0],
#                [2, 0, 0, 0, 0, 0, 2, 0, 0],
#                [2, 0, 0, 0, 0, 0, 2, 0, 0],
#                [2, 0, 0, 2, 0, 0, 2, 0, 0],
#                [2, 0, 0, 0, 0, 0, 2, 0, 0],
#                [2, 0, 0, 0, 0, 0, 2, 0, 0],
#                [2, 2, 2, 2, 2, 2, 2, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0]]

# train_output = [[2, 2, 2, 2, 2, 2, 2, 0, 0],
#                 [2, 4, 4, 4, 4, 4, 2, 0, 0],
#                 [2, 4, 4, 4, 4, 4, 2, 0, 0],
#                 [2, 4, 4, 2, 4, 4, 2, 0, 0],
#                 [2, 4, 4, 4, 4, 4, 2, 0, 0],
#                 [2, 4, 4, 4, 4, 4, 2, 0, 0],
#                 [2, 2, 2, 2, 2, 2, 2, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0]]


# train_input = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0],
#                [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
#                [0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0],
#                [0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
#                [0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0],
#                [0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
#                [0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
#                [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
#                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# train_output = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0],
#                 [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 8, 8, 8, 2, 0, 0, 0, 0],
#                 [0, 2, 3, 3, 3, 3, 3, 3, 3, 2, 0, 2, 8, 2, 8, 2, 0, 0, 0, 0],
#                 [0, 2, 3, 3, 3, 3, 3, 3, 3, 2, 0, 2, 8, 8, 8, 2, 0, 0, 0, 0],
#                 [0, 2, 3, 3, 3, 3, 3, 3, 3, 2, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0],
#                 [0, 2, 3, 3, 3, 2, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 2, 3, 3, 3, 3, 3, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 2, 3, 3, 3, 3, 3, 3, 3, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
#                 [0, 2, 3, 3, 3, 3, 3, 3, 3, 2, 0, 0, 0, 2, 8, 8, 8, 2, 0, 0],
#                 [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 8, 2, 8, 2, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8, 8, 8, 2, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
#                 [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 2, 4, 4, 4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 2, 4, 4, 4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 2, 4, 4, 2, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 2, 4, 4, 4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 2, 4, 4, 4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# SN = SemNet([train_input, train_output])


    
            
    
            
            
        