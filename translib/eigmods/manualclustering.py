#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:00:05 2025

@author: alberto-razo
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib.colors import LogNorm
from PyQt5.QtWidgets import QFileDialog
from tkinter.simpledialog import askfloat
from matplotlib.widgets import PolygonSelector, MultiCursor

class manualClusterClass():
    def __init__(self, HarmonicInversionName, Domain=None, Dimension=None, ScattersPos=None, 
                 MatSize=None, Histograms=None):
        self.HarmonicInversionName = HarmonicInversionName
        self.DOSmode = False
        self.ScattersPos = ScattersPos
        self.Histograms = Histograms
        self.Dimension = Dimension
        self.Domain = Domain
        self.MatSize = MatSize
        self.Dimensions = ['1D', '2D', 'Discrete_1D', 'Discrete_2D']

        if not os.path.isfile(self.HarmonicInversionName) or self.HarmonicInversionName==None:
            print(f'The indicated file {self.HarmonicInversionName} does not exist, please verify it.')
            return None
        elif (self.Domain is None) or (isinstance(self.Domain, Iterable) and self.Domain.any is None):
            print('The spatial domain of the function has not been indicated, please do it.')
            return None
        elif not self.Dimension in self.Dimensions:
            while not self.Dimension in self.Dimensions:
                print('Any dimension indicated, or option not avaliable (Options avaliable: 1D, 2D, Discrete_1D, Discrete_2D)')
                self.Dimension = input('Which is the dimension of the system?')
                
        if 'Discrete' in self.Dimension and ((self.ScattersPos is None) or (isinstance(self.ScattersPos, Iterable) and self.ScattersPos.any is None)):
            print(f'For {self.Dimension}, the scatter/resonator position is needed.')
            return None
        
        if self.MatSize is None and ((self.ScattersPos is not None) or (isinstance(self.ScattersPos, Iterable) and self.ScattersPos.any is not None)):
            if '2D' in self.Dimension:
                self.MatSize = np.float64(input('What is the radius of your scatterers in mm or relative to the total area?'))
            elif '1D' in self.Dimension:
                self.MatSize = np.float64(input('What is the length of your scatterers in mm or relative to the total length?'))

        self.theDict = np.load(self.HarmonicInversionName, allow_pickle=True).item()
        self.theDict["allF0"] = np.array(self.theDict["allF0"])
        self.theDict["allGamma0"] = np.array(self.theDict["allGamma0"])
        self.theDict["allAbs"] = np.array(self.theDict["allAbs"])
        self.theDict["allArg"] = np.array(self.theDict["allArg"])
        if self.Dimension =='1D':
            self.theDict["X"] = np.array(self.theDict["X"])
        elif self.Dimension == '2D':
            self.theDict["X"] = np.array(self.theDict["X"])
            self.theDict["Y"] = np.array(self.theDict["Y"])
            self.theDict["XY"] = np.array(self.theDict["XY"])
        else:
            self.theDict["Index"] = np.array(self.theDict["Index"])
        if self.Histograms:
            self.histzorder = 10
        self.Clusters = []
        
        
    def prepareSelection(self):
        selection = self.theDict["allF0"]
        self._selection = np.intersect1d(np.where(selection>self.minf), np.where(selection<=self.maxf()))
        if self.maxAmpl is not None:
            self._selection = np.intersect1d(self._selection, np.where(self.theDict["allAbs"]<self.maxAmpl))
        if self.minWidth is not None:
            self._selection = np.intersect1d(self._selection, np.where(self.theDict["allGamma0"]>self.minWidth))
        if self.maxWidth is not None:
            self._selection = np.intersect1d(self._selection, np.where(self.theDict["allGamma0"]<self.maxWidth))
        
        self.selFreq = self.theDict["allF0"][self._selection]
        
        
    def maxf(self):
        return self.minf+self.deltaf
    
    
    def mylog(self,ArrayToCompute):
        def _mylog(ArrayToCompute):
            if ArrayToCompute==0:
                return np.nan
            else:
                return np.log(ArrayToCompute)
        return np.vectorize(_mylog)(ArrayToCompute)
        
    
    def doIt(self, minf, deltaf, minWidth=None, maxWidth=None, maxAmpl=None, histGrid=[800,200]):
        print("### Keys ###")
        print("→/←: move the frequency axis")
        print("  f: Change the frequency span")
        print("  w: Change the min/max width")
        print("  a: Change the max amplitude")
        print("  r: remove the last added cluster")
        print("  p: plot the last added cluster")
        print("     You can then navigate through the clusters with →/← in the clustering result window")
        print("  s: Save cluster file")
        print("  l: Load cluster file")
        print("  g: Save the data of the last selected cluster")
        print("  d: Change to manual counting mode")
        print("  i: Defines a new cluster from the intersection of the last two selected clusters")
        print("  u: Defines a new cluster from the union of the last two selected clusters")
        print("  m: Defines a new cluster from the difference of the last selected clusters minus the")
        print("     second last cluster")
        print("############")
        
        self.minWidth = minWidth
        self.maxWidth = maxWidth
        self.maxAmpl = maxAmpl
        self.histGrid = histGrid
        
        self.minf = minf
        self.deltaf = deltaf
        
        self.prepareSelection()
        
        
        if self.Dimension=='2D':
            self.indexAx = 2
        else:
            self.indexAx = 0
            
        fig,ax=plt.subplots(self.indexAx+4, 1, figsize=(8,9), sharex=True)
        self.pltArgsPlot = {"marker": "o", "markersize": np.sqrt(5), "markeredgewidth": 0, "color": "#BDBDBD", "alpha": 0.25, "linestyle": ""}
        self.ax=ax
        self.fig=fig
        plt.xlim((self.minf,self.maxf()))
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        fig.canvas.mpl_connect('key_press_event', lambda event: self.onkey(event))
        
        if self.Dimension=='2D':
            self.f_X=[]
            self.f_Y=[]
            self.f_XY=[]
        elif self.Dimension=='1D':
            self.f_X=[]
        else: 
            self.f_Index=[]
        self.f_Width=[]
        self.f_Arg=[]
        self.f_Abs=[]
        
        if not self.DOSmode:
            if self.Dimension=='2D':
                self.poly_X = PolygonSelector(self.ax[0], self.polygon_select_callback_X)
                self.poly_Y = PolygonSelector(self.ax[1],self.polygon_select_callback_Y)
                self.poly_XY = PolygonSelector(self.ax[2], self.polygon_select_callback_XY)
            elif self.Dimension=='1D':
                self.poly_X = PolygonSelector(self.ax[0], self.polygon_select_callback_X)
            else:
                self.poly_Index = PolygonSelector(self.ax[0],self.polygon_select_callback_Index)
            self.poly_witdth = PolygonSelector(self.ax[self.indexAx+1], self.polygon_select_callback_width)
            self.poly_phase = PolygonSelector(self.ax[self.indexAx+2], self.polygon_select_callback_phase)
            self.poly_amplitude = PolygonSelector(self.ax[self.indexAx+3],self.polygon_select_callback_amplitude)
            
        plotTitle=""
        plt.sca(ax[0])
        plt.title(plotTitle)
        if self.Dimension=='2D':
            self.plot_f_X, = plt.plot(self.selFreq, self.theDict["X"][self._selection], **self.pltArgsPlot)
            plt.ylabel("X")
            
            plt.sca(ax[1])
            self.plot_f_Y, = plt.plot(self.selFreq, self.theDict["Y"][self._selection], **self.pltArgsPlot)
            plt.ylabel("Y")
            
            plt.sca(ax[2])
            self.plot_f_XY, = plt.plot(self.selFreq, self.theDict["XY"][self._selection], **self.pltArgsPlot)
            plt.ylabel("Spectrum number")
        elif self.Dimension=='1D':
            self.plot_f_X, = plt.plot(self.selFreq, self.theDict["X"][self._selection], **self.pltArgsPlot)
            plt.ylabel("X")
        else:
            self.plot_f_Index, = plt.plot(self.selFreq, self.theDict["Index"][self._selection], **self.pltArgsPlot)
            plt.ylabel("Index")
        plt.sca(ax[self.indexAx+1])
        self.plot_f_Gamma0, = plt.plot(self.selFreq, self.theDict["allGamma0"][self._selection], **self.pltArgsPlot)
        plt.ylabel("Width [GHz]")
        plt.ylim((self.minWidth,self.maxWidth))
        
        plt.sca(ax[self.indexAx+2])
        self.plot_f_Arg, = plt.plot(self.selFreq, self.theDict["allArg"][self._selection], **self.pltArgsPlot)
        plt.ylabel("Phase")
        
        plt.sca(ax[self.indexAx+3])
        self.plot_f_Abs, = plt.plot(self.selFreq, np.log10(self.theDict["allAbs"][self._selection]), **self.pltArgsPlot)
        xedges_Abs = np.linspace(self.minf, self.minf+self.deltaf, self.histGrid[0])
        yedges_Abs = np.logspace(-10, np.log10(self.maxAmpl), self.histGrid[1])
        plt.ylabel("log10(Amplitude)")
        plt.xlabel("Frequency [GHz]")
        if self.Histograms:
            plt.sca(ax[self.indexAx+1])
            hist_f_Gamma0_data, xedges, yedges = np.histogram2d(self.selFreq, self.theDict["allGamma0"][self._selection],
                                                                bins=self.histGrid)
            self.hist_f_Gamma0 = plt.imshow(self.mylog(hist_f_Gamma0_data.T), zorder=self.histzorder,
                                           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                           origin='lower', aspect='auto')
            plt.sca(ax[self.indexAx+3])
            hist_f_Abs_data, xedges, yedges = np.histogram2d(self.selFreq, self.theDict["allAbs"][self._selection], bins=(xedges_Abs, yedges_Abs))
            self.hist_f_Abs = plt.imshow(hist_f_Abs_data.T, zorder=self.histzorder, norm=LogNorm(),
                                       extent=[xedges[0], xedges[-1], np.log10(yedges[0]), np.log10(yedges[-1])],
                                       origin='lower', aspect='auto')
        self.setPlotLims()
        plt.tight_layout()
        plt.show()
    
            
    def polygon_select_callback_XY(self, verts):
        p = mpl.path.Path(verts)
        self._localSelection = p.contains_points(np.vstack((self.theDict["allF0"], self.theDict["XY"])).T)
        self._localSelection = np.where(self._localSelection)[0]
        self._localSelection = np.intersect1d(self._localSelection, self._selection)
        self.propagateSelection()
        
        
    def polygon_select_callback_X(self, verts):
        p = mpl.path.Path(verts)
        self._localSelection = p.contains_points(np.vstack((self.theDict["allF0"], self.theDict["X"])).T)
        self._localSelection = np.where(self._localSelection)[0]
        self._localSelection = np.intersect1d(self._localSelection, self._selection)
        self.propagateSelection()
        
        
    def polygon_select_callback_Y(self, verts):
        p = mpl.path.Path(verts)
        self._localSelection = p.contains_points(np.vstack((self.theDict["allF0"], self.theDict["Y"])).T)
        self._localSelection = np.where(self._localSelection)[0]
        self._localSelection = np.intersect1d(self._localSelection, self._selection)
        self.propagateSelection()
        
        
    def polygon_select_callback_Index(self, verts):
        p = mpl.path.Path(verts)
        self._localSelection = p.contains_points(np.vstack((self.theDict["allF0"], self.theDict["Index"])).T)
        self._localSelection = np.where(self._localSelection)[0]
        self._localSelection = np.intersect1d(self._localSelection, self._selection)
        self.propagateSelection()
        
        
    def polygon_select_callback_width(self, verts):
        p = mpl.path.Path(verts)
        self._localSelection = p.contains_points(np.vstack((self.theDict["allF0"], self.theDict["allGamma0"])).T)
        self._localSelection = np.where(self._localSelection)[0]
        self._localSelection = np.intersect1d(self._localSelection, self._selection)
        self.propagateSelection()
        
        
    def polygon_select_callback_phase(self, verts):
        p = mpl.path.Path(verts)
        self._localSelection = p.contains_points(np.vstack((self.theDict["allF0"], self.theDict["allArg"])).T)
        self._localSelection = np.where(self._localSelection)[0]
        self._localSelection = np.intersect1d(self._localSelection, self._selection)
        self.propagateSelection()
        
        
    def polygon_select_callback_amplitude(self, verts):
        logVerts = []
        for i in range(len(verts)):
            a, b = verts[i]
            logVerts.append((a, np.log(10**b)))
        p = mpl.path.Path(logVerts)
        self._localSelection = p.contains_points(np.vstack((self.theDict["allF0"], np.log(self.theDict["allAbs"]))).T)
        self._localSelection = np.where(self._localSelection)[0]
        self._localSelection = np.intersect1d(self._localSelection, self._selection)
        self.propagateSelection()
        
    
    def setPlotLims(self):
        if self.Dimension=='2D':
            minX = np.min(self.theDict["X"])
            maxX = np.max(self.theDict["X"])
            amplX = maxX - minX
            self.ax[0].set_ylim(minX-0.05*amplX, maxX+0.05*amplX)
            
            minY = np.min(self.theDict["Y"])
            maxY = np.max(self.theDict["Y"])
            amplY = maxY - minY
            self.ax[1].set_ylim(minY-0.05*amplY, maxY+0.05*amplY)
            
            minXY = np.min(self.theDict["XY"])
            maxXY = np.max(self.theDict["XY"])
            amplXY = maxXY - minXY
            self.ax[2].set_ylim(minXY-0.05*amplXY, maxXY+0.05*amplXY)
        elif self.Dimension=='1D':   
            minX = np.min(self.theDict["X"])
            maxX = np.max(self.theDict["X"])
            amplX = maxX - minX
            self.ax[0].set_ylim(minX-0.05*amplX, maxX+0.05*amplX)
        else:
            minIndex = np.min(self.theDict["Index"])
            maxIndex = np.max(self.theDict["Index"])
            amplIndex = maxIndex - minIndex
            self.ax[0].set_ylim(minIndex-0.05*amplIndex, maxIndex+0.05*amplIndex)
        
        self.ax[0].set_xlim((self.minf,self.maxf()))
        minW = (self.minWidth if self.minWidth is not None else np.min(self.theDict["allGamma0"]))
        maxW = (self.maxWidth if self.maxWidth is not None else np.max(self.theDict["allGamma0"]))
        amplW = maxW - minW
        self.ax[self.indexAx+1].set_ylim(minW-0.05*amplW, maxW+0.05*amplW)
        self.ax[self.indexAx+2].set_ylim(-1.1*np.pi, 1.1*np.pi)
        minA = np.min(self.theDict["allAbs"])
        maxA = (self.maxAmpl if self.maxAmpl is not None else np.max(self.theDict["allAbs"]))
        self.ax[self.indexAx+3].set_ylim(np.log10(0.5*minA), np.log10(2*maxA))
        
        
    def propagateSelection(self):
        self.Clusters.append(self._localSelection)
        pltArgs={"s": 5, "linewidth": 0, "alpha": 0.35, "zorder": 100}
        
        if self.Dimension == '2D':
            plt.sca(self.ax[0])
            self.f_X.append(plt.scatter(self.theDict["allF0"][self._localSelection], self.theDict["X"][self._localSelection], **pltArgs))
            plt.sca(self.ax[1])
            self.f_Y.append(plt.scatter(self.theDict["allF0"][self._localSelection], self.theDict["Y"][self._localSelection], **pltArgs))         
            plt.sca(self.ax[2])
            self.f_XY.append(plt.scatter(self.theDict["allF0"][self._localSelection], self.theDict["XY"][self._localSelection], **pltArgs))
        elif self.Dimension == '1D':
            plt.sca(self.ax[0])
            self.f_X.append(plt.scatter(self.theDict["allF0"][self._localSelection], self.theDict["X"][self._localSelection], **pltArgs))
        else:
            plt.sca(self.ax[0])
            self.f_Index.append(plt.scatter(self.theDict["allF0"][self._localSelection], self.theDict["Index"][self._localSelection], **pltArgs))
            
        plt.sca(self.ax[self.indexAx+1])
        self.f_Width.append(plt.scatter(self.theDict["allF0"][self._localSelection], self.theDict["allGamma0"][self._localSelection], **pltArgs))
        
        plt.sca(self.ax[self.indexAx+2])
        self.f_Arg.append(plt.scatter(self.theDict["allF0"][self._localSelection], self.theDict["allArg"][self._localSelection], **pltArgs))
        
        plt.sca(self.ax[self.indexAx+3])
        self.f_Abs.append(plt.scatter(self.theDict["allF0"][self._localSelection],
                                      np.log10(self.theDict["allAbs"][self._localSelection]), **pltArgs))
        self.fig.canvas.draw()
        
        
    def onkey(self, event):
        def askParameter(prompt):
            parameter = None
            while parameter is None:
                parameter = askfloat("Input", f'{prompt}:')
                if parameter is None:
                    print('No valid input, please write a valid value')
            return parameter
        
        def RemoveLastTwoClusters(_ind):
            self.Clusters[-2] = _ind
            self.Clusters.pop(-1)
            for i in range(2):
                if self.Dimension == '2D':
                    self.f_X[-1].remove()
                    del self.f_X[-1]
                    self.f_Y[-1].remove()
                    del self.f_Y[-1]
                    self.f_XY[-1].remove()
                    del self.f_XY[-1]
                elif self.Dimension == '1D':
                    self.f_X[-1].remove()
                    del self.f_X[-1]
                else:
                    self.f_Index[-1].remove()
                    del self.f_Index[-1]
                
                self.f_Width[-1].remove()
                del self.f_Width[-1]
                self.f_Arg[-1].remove()
                del self.f_Arg[-1]
                self.f_Abs[-1].remove()
                del self.f_Abs[-1]
            
            self._localSelection = _ind
            pltArgs = {"s": 5, "linewidth": 0, "alpha": 0.35, "zorder": 10} 
            if self.Dimension == '2D':
                plt.sca(self.ax[0])
                self.f_X.append(plt.scatter(self.theDict["allF0"][self._localSelection], self.theDict["X"][self._localSelection], **pltArgs))
                plt.sca(self.ax[1])
                self.f_Y.append(plt.scatter(self.theDict["allF0"][self._localSelection], self.theDict["Y"][self._localSelection], **pltArgs))
                plt.sca(self.ax[2])
                self.f_XY.append(plt.scatter(self.theDict["allF0"][self._localSelection], self.theDict["XY"][self._localSelection], **pltArgs))
            elif self.Dimension == '1D':
                plt.sca(self.ax[0])
                self.f_X.append(plt.scatter(self.theDict["allF0"][self._localSelection], self.theDict["X"][self._localSelection], **pltArgs))
            else:
                plt.sca(self.ax[0])
                self.f_Index.append(plt.scatter(self.theDict["allF0"][self._localSelection], self.theDict["Index"][self._localSelection], **pltArgs))
            
            plt.sca(self.ax[self.indexAx+1])
            self.f_Width.append(plt.scatter(self.theDict["allF0"][self._localSelection],self.theDict["allGamma0"][self._localSelection],**pltArgs))
            plt.sca(self.ax[self.indexAx+2])
            self.f_Arg.append(plt.scatter(self.theDict["allF0"][self._localSelection],self.theDict["allArg"][self._localSelection],**pltArgs))
            plt.sca(self.ax[self.indexAx+3])
            self.f_Abs.append(plt.scatter(self.theDict["allF0"][self._localSelection],
                                          np.log10(self.theDict["allAbs"][self._localSelection]),**pltArgs))
            
        if event.key == "right":
            self.minf += self.deltaf/2
            
        elif event.key == "left":
            self.minf -= self.deltaf/2
            
        elif event.key == "a":
            self.maxAmpl = askParameter("Maximum Amplitude")
            if self.maxAmpl < 0:
                self.maxAmpl = None
                
        elif event.key == "w":
            self.minWidth = askParameter("Minimum Width")
            if self.minWidth < 0:
                self.minWidth = None
            self.maxWidth = askParameter("Maximum Width")
            if self.maxWidth < 0:
                self.maxWidth = None
                
        elif event.key == "f":
            deltaf = askParameter("Frequency span (GHz)")
            if deltaf > 0:
                self.deltaf = deltaf
                
        elif event.key == "r":
            if not self.DOSmode:
                self.Clusters.pop(-1)
                if self.Dimension == '2D':
                    self.f_X[-1].remove()
                    del self.f_X[-1]
                    self.f_Y[-1].remove()
                    del self.f_Y[-1]
                    self.f_XY[-1].remove()
                    del self.f_XY[-1]
                elif self.Dimension == '1D':
                    self.f_X[-1].remove()
                    del self.f_X[-1]
                else:
                    self.f_Index[-1].remove()
                    del self.f_Index[-1]
                    
                self.f_Width[-1].remove()
                del self.f_Width[-1]
    
                self.f_Arg[-1].remove()
                del self.f_Arg[-1]
    
                self.f_Abs[-1].remove()
                del self.f_Abs[-1]
            else:
                self.ax[0].lines[-1].remove()
                if self.Dimension == '2D':
                    self.ax[1].lines[-1].remove()
                    self.ax[2].lines[-1].remove()
                self.ax[self.indexAx+1].lines[-1].remove()
                self.ax[self.indexAx+2].lines[-1].remove()
                self.ax[self.indexAx+3].lines[-1].remove()
                
                self.LastClick[-1].collections[-1].remove()
                self.LastClick=self.LastClick[:-1]
                self.Frequencies=self.Frequencies[:-1]

                self.fig.canvas.draw()
                
        elif event.key == "i":
            if not self.DOSmode:
                _Cluster1 = self.Clusters[-1]
                _Cluster2 = self.Clusters[-2]
                _ind = list(set(_Cluster1) & set(_Cluster2))
                
                RemoveLastTwoClusters(_ind)
                self.fig.canvas.draw()
                
        elif event.key == "u":
            if not self.DOSmode:
                _Cluster1=self.Clusters[-1]
                _Cluster2=self.Clusters[-2]
                _ind = list(set(_Cluster1) | set(_Cluster2))
                
                RemoveLastTwoClusters(_ind)
                self.fig.canvas.draw()
                
        elif event.key=="m":
            if not self.DOSmode:
                _Cluster1 = self.Clusters[-1]
                _Cluster2 = self.Clusters[-2]
                _ind = list(set(_Cluster2) - set(_Cluster1))
                
                RemoveLastTwoClusters(_ind)
                self.fig.canvas.draw()
                
        elif event.key == "h":
            if self.Histograms == True:
                self.histzorder = -self.histzorder
            else:
                print('The histogram option is not activated')
                
        elif event.key == "s":
            filename, _ = QFileDialog.getSaveFileName(None, "Select clusters file", '', 'npy files (*.npy);;all files (*.*)')
            _theDict = {}
            _theDict["theClusters"] = self.Clusters
            np.save(filename, _theDict, allow_pickle=True)
            print(f'File {filename} with {len(self.Clusters)} cluster(s) saved')
            
        elif event.key == "l":
            filename, _ = QFileDialog.getOpenFileName(None, 'Select clusters file', '', 'npy files (*.npy);;all files (*.*)')
            _theDict = np.load(filename, allow_pickle=True).item()
            for Cluster in _theDict["theClusters"]:
                self._localSelection = Cluster
                self.propagateSelection()
            print(f'File {filename} with {len(_theDict["theClusters"])} cluster(s) loaded')
            
        elif event.key == "g":
            self.plottedClusterIndex = -1
            Cluster = self.Clusters[self.plottedClusterIndex]
            self.ClusterTheData(Cluster)
            
            filename, _ = QFileDialog.getSaveFileName(None, "Select mode file", '', 'npy files (*.npy);;all files (*.*)')
            _theDict = {}
            _theDict['Mode'] = self.ModeMap
            _theDict['Qfactor'] = self.Qfactor
            _theDict['MeanFreq'] = self.meanFreq
            _theDict['IPR'] = self.IPR
            _theDict['Complexness'] = self.complexness
            _theDict['PhaseRigidity'] = self.phaseRigidity
            np.save(filename, _theDict, allow_pickle=True)
            print(f'Mode with Frq={self.meanFreq} has been saved as {filename}')
            
        elif event.key == "p":
            self.figClusteringResult = plt.figure(figsize=(15,10))
            self.figClusteringResult.canvas.mpl_connect('key_press_event', lambda event: self.onkeyClusteringResult(event))
            self.plottedClusterIndex = -1 #we start with the last one
            self.plotClusteringResult()
            plt.show()
                
        elif event.key == "d":
            self.DOSmode = not self.DOSmode
            if self.DOSmode:
                self.Frequencies = []
                self.Width = []
                self.LastClick = []
                
                if self.Dimension == '2D':
                    del self.poly_X
                    del self.poly_Y
                    del self.poly_XY
                elif self.Dimension == '1D':
                    del self.poly_X
                else:
                    del self.poly_Index
                    
                del self.poly_witdth
                del self.poly_phase
                del self.poly_amplitude
                
                def MouseClick(event):
                    if event.inaxes is not None:
                        print(f'Frequency={event.xdata} GHz, Width={event.ydata} GHz')
                        event.inaxes.scatter([event.xdata],[event.ydata], zorder=1000, color='r')
                        self.LastClick = np.append(self.LastClick, event.inaxes)
                        self.Frequencies = np.append(self.Frequencies, event.xdata)
                        self.Width = np.append(self.Width,event.ydata)
                        
                        self.ax[0].axvline(event.xdata, color='r', lw=1, zorder=25)
                        if self.Dimension == '2D':
                            self.ax[1].axvline(event.xdata, color='r', lw=1, zorder=25)
                            self.ax[2].axvline(event.xdata, color='r', lw=1, zorder=25)

                        self.ax[self.indexAx+1].axvline(event.xdata,color='r',lw=1,zorder=25)
                        self.ax[self.indexAx+2].axvline(event.xdata,color='r',lw=1,zorder=25)
                        self.ax[self.indexAx+3].axvline(event.xdata,color='r',lw=1,zorder=25)
                    try:
                        self.fig.canvas.draw()
                    except:
                        pass
                    
                def ModeCounting():
                    self.RedSelectingLine = MultiCursor(self.fig.canvas, self.ax, color='r',lw=1)
                    self.cid=self.fig.canvas.mpl_connect('button_press_event', MouseClick)

                ModeCounting()

            else:
                self.fig.canvas.mpl_disconnect(self.cid)
                del self.RedSelectingLine
                
                if self.Dimension == '2D':
                    self.poly_X = PolygonSelector(self.ax[0], self.polygon_select_callback_X)
                    self.poly_Y = PolygonSelector(self.ax[1], self.polygon_select_callback_Y)
                    self.poly_XY = PolygonSelector(self.ax[2], self.polygon_select_callback_XY)
                elif self.Dimension == '1D':
                    self.poly_X = PolygonSelector(self.ax[0], self.polygon_select_callback_X)
                else:
                    self.poly_Index = PolygonSelector(self.ax[0], self.polygon_select_callback_Index)
                
                self.poly_witdth = PolygonSelector(self.ax[self.indexAx+1], self.polygon_select_callback_width)
                self.poly_phase = PolygonSelector(self.ax[self.indexAx+2], self.polygon_select_callback_phase)
                self.poly_amplitude = PolygonSelector(self.ax[self.indexAx+3],self.polygon_select_callback_amplitude)
                
                filename, _ = QFileDialog.getSaveFileName(None, "Select DoS file", '', 'npy files (*.npy);;all files (*.*)')
                dataToSave = np.zeros((len(self.Frequencies), 2))
                dataToSave[:,0] = self.Frequencies
                dataToSave[:,1] = self.Width
                np.save(filename, dataToSave, allow_pickle=True)
                
        self.setPlotLims()

        self.prepareSelection()
        
        if self.Dimension == '2D':
            self.plot_f_X.set_data(self.selFreq, self.theDict["X"][self._selection])
            self.plot_f_Y.set_data(self.selFreq, self.theDict["Y"][self._selection])
            self.plot_f_XY.set_data(self.selFreq, self.theDict["XY"][self._selection])
        elif self.Dimension == '1D':
            self.plot_f_X.set_data(self.selFreq, self.theDict["X"][self._selection])
        else:
            self.plot_f_Index.set_data(self.selFreq, self.theDict["Index"][self._selection])
            
        self.plot_f_Gamma0.set_data(self.selFreq, self.theDict["allGamma0"][self._selection])
        self.plot_f_Arg.set_data(self.selFreq, self.theDict["allArg"][self._selection])
        self.plot_f_Abs.set_data(self.selFreq, np.log10(self.theDict["allAbs"][self._selection]))
        
        if self.Histograms == True:
            hist_f_Gamma0_data, xedges, yedges = np.histogram2d(self.selFreq, self.theDict["allGamma0"][self._selection],
                                                              bins=self.histGrid)
            self.hist_f_Gamma0.set_data(self.mylog(hist_f_Gamma0_data.T))
            self.hist_f_Gamma0.set_extent([xedges[0], xedges[-1], yedges[0], yedges[-1]])
            self.hist_f_Gamma0.set_zorder(self.histzorder)
            
            xedges_Abs = np.linspace(self.maxf()-self.deltaf, self.maxf(),self.histGrid[0])
            yedges_Abs = np.logspace(-10, np.log10(self.maxAmpl), self.histGrid[1])
            
            hist_f_Abs_data, xedges, yedges = np.histogram2d(self.selFreq, self.theDict["allAbs"][self._selection], bins=(xedges_Abs, yedges_Abs))
            self.hist_f_Abs.set_data(hist_f_Abs_data.T)
            self.hist_f_Abs.set_extent([xedges[0], xedges[-1], np.log10(yedges[0]), np.log10(yedges[-1])])
            self.hist_f_Abs.set_zorder(self.histzorder)
        
        try:
            self.fig.canvas.draw()
        except:
            pass
        
        
    def createFigureAxis(self):
        self.figClusteringResult.clear()
        
        _Spec0 = mpl.gridspec.GridSpec(ncols=4, nrows=4, wspace=0.4, hspace=0.3)
        
        if self.Dimension == '2D':
            _Spec1 = mpl.gridspec.GridSpec(ncols=4, nrows=3, wspace=0.55, hspace=0.25, width_ratios=[1,1.25,1.25,1])
        elif self.Dimension == 'Discrete_2D':
            _Spec1 = mpl.gridspec.GridSpec(ncols=6, nrows=3, wspace=0.55, hspace=0.25, width_ratios=[1.2, 1, 0.03, 1, 0.03, 1])
            _Spec2 = mpl.gridspec.GridSpec(ncols=6, nrows=3, wspace=0, hspace=0.25, width_ratios=[0.8, 0.7, 0.03, 1, 0.03, 1])
        else:
            _Spec1 = mpl.gridspec.GridSpec(ncols=3, nrows=5, wspace=0.3, hspace=0.3, width_ratios=[1,2,1], height_ratios=[1.5,1,1,1,1])
            
        self._ax = []
        for i in range(4):
            self._ax.append(self.figClusteringResult.add_subplot(_Spec0[4*i]))
        if self.Dimension == '2D':
            self._ax.append(self.figClusteringResult.add_subplot(_Spec1[1], aspect=1))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec1[2], aspect=1))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec1[5], aspect=1))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec1[6], aspect=1))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec1[9], aspect=1))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec1[10], aspect=1))
        elif self.Dimension == 'Discrete_2D':
            self._ax.append(self.figClusteringResult.add_subplot(_Spec1[1], aspect=1))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec2[2]))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec1[3], aspect=1))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec1[7], aspect=1))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec2[8]))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec1[9], aspect=1))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec2[10]))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec1[13], aspect=1))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec2[14]))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec1[15], aspect=1))
            self._ax.append(self.figClusteringResult.add_subplot(_Spec2[16]))
        else:
            self._ax.append(self.figClusteringResult.add_subplot(_Spec0[1]))    
            self._ax.append(self.figClusteringResult.add_subplot(_Spec0[2], aspect=1))
            for i in range(4):
                self._ax.append(self.figClusteringResult.add_subplot(_Spec1[3*(i+1)+1]))

        for i in range(4):
            self._ax.append(self.figClusteringResult.add_subplot(_Spec0[4*i+3]))
            
        self.figClusteringResult.suptitle(rf'Frq = {self.meanFreq:0.3f} GHz, IPR = {self.IPR:03f}'+'\n'+'→/←: move the frequency axis')
        
        
    def plotClusteringResult(self):
        
        Cluster = self.Clusters[self.plottedClusterIndex]
        self.ClusterTheData(Cluster)
        
        self.createFigureAxis()
        
        self.plotCluster(self._ax[:4])
        if self.Dimension == 'Discrete_2D':
            self.plotUnicityMap(self._ax[4:6])
            self.plotPhaseRotation(self._ax[6])
            self.plotMap(self._ax[7:9])
            self.plotPhaseMap(self._ax[9:11])
            self.plotFreqMap(self._ax[11:13])
            self.plotWidthMap(self._ax[13:15])
            self.plotIntHist(self._ax[15])
            self.plotPhaseHist(self._ax[16])
            self.plotFreqHist(self._ax[17])
            self.plotWidthHist(self._ax[18])
        else:
            self.plotUnicityMap(self._ax[4])
            self.plotPhaseRotation(self._ax[5])
            self.plotMap(self._ax[6])
            self.plotPhaseMap(self._ax[7])
            self.plotFreqMap(self._ax[8])
            self.plotWidthMap(self._ax[9])
            self.plotIntHist(self._ax[10])
            self.plotPhaseHist(self._ax[11])
            self.plotFreqHist(self._ax[12])
            self.plotWidthHist(self._ax[13])
        
        self.figClusteringResult.canvas.draw()
        

    def onkeyClusteringResult(self,event):
        if event.key == "right":
            print("Plotting next cluster")
            self.plottedClusterIndex += 1
        elif event.key == "left":
            print("Plotting previous cluster")
            self.plottedClusterIndex -= 1
        self.plottedClusterIndex = self.plottedClusterIndex%len(self.Clusters)
        self.plotClusteringResult()
        
        
    def ClusterTheData(self, Cluster):
        self.ampl = self.theDict["allAbs"][Cluster]
        self.phase = self.theDict["allArg"][Cluster]
        self.freq = self.theDict["allF0"][Cluster]
        self.width = self.theDict["allGamma0"][Cluster]
        
        self.complexAmplitude = self.ampl*np.exp(1j*self.phase)
        self.mode_Re = self.complexAmplitude.real
        self.mode_Im = self.complexAmplitude.imag
        self.tan2alpha = -2*np.mean(self.mode_Re*self.mode_Im)/(np.mean((self.mode_Re)**2)-np.mean((self.mode_Im)**2))
        self.alpha = np.arctan(self.tan2alpha)/2
        self.modePrime = (self.mode_Re+1j*self.mode_Im)*np.exp(1j*(self.alpha))
        if np.mean(np.abs(self.modePrime.real))<np.mean(np.abs(self.modePrime.imag)): #turned cigar is vertical
            self.alpha += np.pi/2
            self.modePrime = (self.mode_Re+1j*self.mode_Im)*np.exp(1j*(self.alpha))
        
        if self.Dimension == '2D':
            self.X = self.theDict["X"][Cluster]
            self.Y = self.theDict["Y"][Cluster]
            self.XY = self.theDict["XY"][Cluster]
            self.EmptyMaps()
            for i in range(len(self.X)):
                _x, _y = self.X[i], self.Y[i]
                indexX = np.where(np.abs(self.Domain-_x)<1E-6)[0][0]
                indexY = np.where(np.abs(self.Domain-_y)<1E-6)[0][0]
                self.unicityMap[indexY, indexX] += 1
                self.ModeMap[indexY, indexX] = np.real(self.modePrime[i])
                self.PhaseMap[indexY, indexX] = np.angle(self.modePrime[i])
                self.FrqMap[indexY, indexX] = self.freq[i]
                self.WidthMap[indexY, indexX] = self.width[i]
        elif self.Dimension == '1D':
            self.X = self.theDict["X"][Cluster]
            self.EmptyMaps()
            for i in range(len(self.X)):
                _x = self.X[i]
                indexX = np.where(np.abs(self.Domain-_x)<1E-6)[0][0]
                self.unicityMap[indexX] += 1
                self.ModeMap[indexX] = np.real(self.modePrime[i])
                self.PhaseMap[indexX] = np.angle(self.modePrime[i])
                self.FrqMap[indexX] = self.freq[i]
                self.WidthMap[indexX] = self.width[i]
        else:
            self.Index = self.theDict["Index"][Cluster]
            self.EmptyMaps()
            for i in range(len(self.Index)):
                _index = self.Index[i]
                indexI = np.where(np.abs(self.Domain-_index)<1E-6)[0][0]
                self.unicityMap[indexI] += 1
                self.ModeMap[indexI] = np.real(self.modePrime[i])
                self.PhaseMap[indexI] = np.angle(self.modePrime[i])
                self.FrqMap[indexI] = self.freq[i]
                self.WidthMap[indexI] = self.width[i]
        
        self.Qfactor = np.mean(self.freq/self.width)
        self.meanFreq = np.mean(self.freq)
        self.IPR = sum(self.ampl**4)/(sum(self.ampl**2))**2
        # Complexness parameter (eq. 1 from Xeridat et al., PRE 80, 035201(R) (2009))
        self.complexness = np.nanmean(self.modePrime.imag**2)/np.nanmean(self.modePrime.real**2)
        # phase rigidity (eq. 15 of PRE 93, 032108 (2016))
        self.phaseRigidity = (1 - self.complexness)/(1 + self.complexness)

        
    def EmptyMaps(self):
        if self.Dimension == '2D':
            self.unicityMap = np.zeros([len(self.Domain), len(self.Domain)], dtype="int")
            self.ModeMap = np.zeros([len(self.Domain), len(self.Domain)])
            self.ModeMap.fill(np.nan)
            self.PhaseMap = np.zeros([len(self.Domain), len(self.Domain)])
            self.PhaseMap.fill(np.nan)
            self.FrqMap = np.zeros([len(self.Domain), len(self.Domain)])
            self.FrqMap.fill(np.nan)
            self.WidthMap = np.zeros([len(self.Domain), len(self.Domain)])
            self.WidthMap.fill(np.nan)
        else:
            self.unicityMap = np.zeros([len(self.Domain)], dtype="int")
            self.ModeMap = np.zeros([len(self.Domain)])
            self.ModeMap.fill(np.nan)
            self.PhaseMap = np.zeros([len(self.Domain)])
            self.PhaseMap.fill(np.nan)
            self.FrqMap = np.zeros([len(self.Domain)])
            self.FrqMap.fill(np.nan)
            self.WidthMap = np.zeros([len(self.Domain)])
            self.WidthMap.fill(np.nan)
        
              
    def plotCluster(self, _axs):
        selection = self.theDict["allF0"]
        selection = np.logical_and(selection>self.meanFreq-0.05, selection<self.meanFreq+0.05)
        
        if self.Dimension == '2D':
            all_XY = self.theDict["XY"][selection]
        elif self.Dimension == '1D':
            all_X = self.theDict["X"][selection]
        else:
            all_Index = self.theDict["Index"][selection]
        all_ampl = self.theDict["allAbs"][selection]
        all_phase = self.theDict["allArg"][selection]
        all_freq = self.theDict["allF0"][selection]
        all_width = self.theDict["allGamma0"][selection]
        
        if self.Dimension == '2D':
            _axs[0].scatter(all_freq, all_XY, s=0.5, c="gray", alpha=0.25, linewidths=0)
            _axs[0].scatter(self.freq, self.XY, s=1, alpha=0.7, linewidths=0)
            _axs[0].set_ylabel("XY")
        elif self.Dimension == '1D':
            _axs[0].scatter(all_freq, all_X, s=0.5, c="gray", alpha=0.25, linewidths=0)
            _axs[0].scatter(self.freq, self.X, s=1, alpha=0.7, linewidths=0)
            _axs[0].set_ylabel("X")
        else:
            _axs[0].scatter(all_freq, all_Index, s=0.5, c="gray", alpha=0.25, linewidths=0)
            _axs[0].scatter(self.freq, self.Index, s=1, alpha=0.7, linewidths=0)
            _axs[0].set_ylabel("Index")
        
        percentile=np.percentile(all_width, 99)
        _axs[1].set_ylim((0, percentile))
        _axs[1].scatter(all_freq, all_width, s=0.5, c="gray", alpha=0.25, linewidths=0)
        _axs[1].scatter(self.freq, self.width, s=1, alpha=0.7, linewidths=0)
        _axs[1].set_ylabel("Width [GHz]")
        
        _axs[2].scatter(all_freq, all_phase, s=0.5, c="gray", alpha=0.25, linewidths=0)
        _axs[2].scatter(self.freq, self.phase, s=1, alpha=0.7, linewidths=0)
        _axs[2].set_ylabel("Phase")
        
        _axs[3].scatter(all_freq, all_ampl, s=0.5, c="gray", alpha=0.25, linewidths=0)
        _axs[3].scatter(self.freq, self.ampl, s=1, alpha=0.7, linewidths=0)
        _axs[3].set_xlabel("Frequency [GHz]")
        _axs[3].set_ylabel("Amplitude")
        _axs[3].set_yscale("log")
        
        
    def plotUnicityMap(self, _ax):
        if self.Dimension == '2D':
            cmap = mpl.cm.viridis
            cmap.set_bad('white',1.)
            bounds = np.array([0, 0.5, 1.5, 2.5, 3])
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            if (self.ScattersPos is not None) or (isinstance(self.ScattersPos, Iterable) and self.ScattersPos.all is not None):
                self.placeScatters(_ax)
            theColorPlot = _ax.imshow(self.unicityMap, origin='lower', 
                                     extent=[self.Domain[0], self.Domain[-1], self.Domain[0], self.Domain[-1]], 
                                     interpolation=None, cmap=cmap, norm=norm)
            cbar = plt.colorbar(theColorPlot, ax=_ax, ticks=[0, 1, 2, 3])
            cbar.ax.set_yticklabels(["0", "1", "2", ">3"])  # vertically oriented colorbar
            _ax.set_title('Unicity map')
            _ax.set_ylabel("y [mm]")
            _ax.set_xlabel("x [mm]")
        elif self.Dimension == '1D':
            _ind = np.where(self.unicityMap > 3)[0]
            self.unicityMap[_ind] = 3
            if (self.ScattersPos is not None) or (isinstance(self.ScattersPos, Iterable) and self.ScattersPos.all is not None):
                self.placeBarriers(_ax)
            _ax.plot(self.Domain, self.unicityMap)
            _ax.set_xlim(self.Domain[0], self.Domain[-1])
            _ax.set_ylim(-0.1, 3.1)
            _ax.set_yticks([0, 1, 2, 3]) 
            _ax.set_yticklabels(["0", "1", "2", ">3"])
            _ax.set_xlabel("x [mm]")
            _ax.set_ylabel('Unicity map')
        elif self.Dimension == 'Discrete_1D':
            _ind = np.where(self.unicityMap > 3)[0]
            self.unicityMap[_ind] = 3
            self.placeBarriers(_ax)
            _ax.plot(self.ScattersPos+self.MatSize/2, self.unicityMap)
            _ax.set_xlim(np.min(self.ScattersPos)-self.MatSize, np.max(self.ScattersPos)+self.MatSize)
            _ax.set_ylim(-0.1, 3.1)
            _ax.set_yticks([0, 1, 2, 3]) 
            _ax.set_yticklabels(["0", "1", "2", ">3"])
            _ax.set_xlabel("x [mm]")
            _ax.set_ylabel('Unicity map')
        elif self.Dimension == 'Discrete_2D':
            cmap = mpl.cm.viridis
            bounds = np.array([0, 0.5, 1.5, 2.5, 3])/3
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            _amps = self.unicityMap/3
            for _i,_cyl in enumerate(self.ScattersPos):
                _circle = plt.Circle((_cyl), self.MatSize, color=cmap(_amps[_i]), lw=0)
                _ax[0].add_artist(_circle) 
            cbar = mpl.colorbar.ColorbarBase(_ax[1], cmap=cmap, ticks=[0, 1/3, 2/3, 1], norm=norm)
            cbar.ax.set_yticklabels(["0", "1", "2", ">3"])
            _ax[0].set_title('Unicity map')
            _ax[0].set_ylabel("y [mm]")
            _ax[0].set_xlabel("x [mm]")
            _ax[0].set_xlim(np.min(self.ScattersPos[:,0])-self.MatSize-5, np.max(self.ScattersPos[:,0])+self.MatSize+5)
            _ax[0].set_ylim(np.min(self.ScattersPos[:,1])-self.MatSize-5, np.max(self.ScattersPos[:,1])+self.MatSize+5)
            
            
    def plotPhaseRotation(self, _ax):
        _ax.scatter(self.mode_Re, self.mode_Im, s=5, alpha=0.5, linewidths=0)
        _ax.scatter(self.modePrime.real, self.modePrime.imag, s=5, alpha=0.5, linewidths=0, label=rf'$\alpha={self.alpha:0.2f}$ rad')
        Vmin = np.min([np.min(self.mode_Re), np.min(self.modePrime.real), np.min(self.mode_Im), np.min(self.modePrime.imag)])
        Vmax = np.max([np.max(self.mode_Re), np.max(self.modePrime.real), np.max(self.mode_Im), np.max(self.modePrime.imag)])
        _ax.set_xlim(Vmin, Vmax)
        _ax.set_xlim(Vmin, Vmax)
        _ax.set_xlabel("Re")
        _ax.set_ylabel("Im")
        _ax.legend()
        
        
    def plotMap(self, _ax):
        if self.Dimension == '2D':
            cmap = mpl.cm.RdBu
            cmap.set_bad('white', 1.)
            if (self.ScattersPos is not None) or (isinstance(self.ScattersPos, Iterable) and self.ScattersPos.all is not None):
                self.placeScatters(_ax)
            theColorPlot = _ax.imshow(self.ModeMap, origin='lower', 
                                     extent=[self.Domain[0], self.Domain[-1], self.Domain[0], self.Domain[-1]],
                                     interpolation=None, cmap=cmap)
            normalization = np.percentile(np.abs(self.ampl), 99)
            theColorPlot.set_clim(vmin=-normalization, vmax=normalization)
            plt.colorbar(theColorPlot, ax=_ax)
            _ax.set_title(r'$\psi$')
            _ax.set_ylabel("y [mm]")
            _ax.set_xlabel("x [mm]")
        elif self.Dimension == '1D':
            if (self.ScattersPos is not None) or (isinstance(self.ScattersPos, Iterable) and self.ScattersPos.all is not None):
                self.placeBarriers(_ax)
            _ax.plot(self.Domain, self.ModeMap)
            _ax.set_xlim(self.Domain[0], self.Domain[-1])
            _ax.set_ylabel(r'$\psi$')
        elif self.Dimension == 'Discrete_1D':
            self.placeBarriers(_ax)
            _ax.plot(self.ScattersPos+self.MatSize/2, self.ModeMap)
            _ax.set_xlim(np.min(self.ScattersPos)-self.MatSize, np.max(self.ScattersPos)+self.MatSize)
            _ax.set_ylabel(r'$\psi$')
        elif self.Dimension == 'Discrete_2D':
            cmap = mpl.cm.RdBu
            normalization = np.percentile(np.abs(self.ampl), 99)
            _amps = (self.ModeMap+normalization)/(2*normalization)
            for _i,_cyl in enumerate(self.ScattersPos):
                _circle = plt.Circle((_cyl), self.MatSize, color=cmap(_amps[_i]), lw=0)
                _ax[0].add_artist(_circle)
            cb = mpl.colorbar.ColorbarBase(_ax[1], cmap = cmap, ticks=[0,0.25,0.5,0.75,1])
            cb.ax.set_yticklabels([f'{-normalization:0.3f}', f'{-normalization*0.5:0.3f}', '0',
                                   f'{normalization*0.5:0.3f}', f'{-normalization:0.3f}'])
            _ax[0].set_title(r'$\psi$')
            _ax[0].set_ylabel("y [mm]")
            _ax[0].set_xlabel("x [mm]")
            _ax[0].set_xlim(np.min(self.ScattersPos[:,0])-self.MatSize-5, np.max(self.ScattersPos[:,0])+self.MatSize+5)
            _ax[0].set_ylim(np.min(self.ScattersPos[:,1])-self.MatSize-5, np.max(self.ScattersPos[:,1])+self.MatSize+5)
            
            
    def plotPhaseMap(self, _ax):
        ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        tickslabels = [r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"]
        if self.Dimension == '2D':
            cmap = mpl.cm.hsv
            cmap.set_bad('white',1.)
            if (self.ScattersPos is not None) or (isinstance(self.ScattersPos, Iterable) and self.ScattersPos.all is not None):
                self.placeScatters(_ax)
            theColorPlot = _ax.imshow(self.PhaseMap, origin='lower', 
                                     extent=[self.Domain[0], self.Domain[-1], self.Domain[0],self.Domain[-1]],
                                     interpolation=None, vmin=-np.pi, vmax=np.pi, cmap=cmap)
            cbar = plt.colorbar(theColorPlot, ax=_ax, ticks=ticks)
            cbar.ax.set_yticklabels(tickslabels)
            _ax.set_title('Phase')
            _ax.set_ylabel("y [mm]")
            _ax.set_xlabel("x [mm]")
        elif self.Dimension == '1D':
            if (self.ScattersPos is not None) or (isinstance(self.ScattersPos, Iterable) and self.ScattersPos.all is not None):
                self.placeBarriers(_ax)
            _ax.plot(self.Domain, self.PhaseMap)
            _ax.set_xlim(self.Domain[0], self.Domain[-1])
            _ax.set_ylabel('Phase')
            _ax.set_yticks(ticks)
            _ax.set_yticklabels(tickslabels)
        elif self.Dimension == 'Discrete_1D':
            self.placeBarriers(_ax)
            _ax.plot(self.ScattersPos+self.MatSize/2, self.PhaseMap)
            _ax.set_xlim(np.min(self.ScattersPos)-self.MatSize, np.max(self.ScattersPos)+self.MatSize)
            _ax.set_ylabel('Phase')
            _ax.set_yticks(ticks)
            _ax.set_yticklabels(tickslabels)
        elif self.Dimension == 'Discrete_2D':
            cmap = mpl.cm.hsv
            _amps = (self.PhaseMap+np.pi)/(2*np.pi)
            for _i,_cyl in enumerate(self.ScattersPos):
                _circle = plt.Circle((_cyl), self.MatSize, color=cmap(_amps[_i]), lw=0)
                _ax[0].add_artist(_circle)
            cb = mpl.colorbar.ColorbarBase(_ax[1], cmap = cmap, ticks=[0, 0.25, 0.5, 0.75, 1])
            cb.ax.set_yticklabels(tickslabels)
            _ax[0].set_title('Phase')
            _ax[0].set_ylabel("y [mm]")
            _ax[0].set_xlabel("x [mm]")
            _ax[0].set_xlim(np.min(self.ScattersPos[:,0])-self.MatSize-5, np.max(self.ScattersPos[:,0])+self.MatSize+5)
            _ax[0].set_ylim(np.min(self.ScattersPos[:,1])-self.MatSize-5, np.max(self.ScattersPos[:,1])+self.MatSize+5)
            
            
    def plotFreqMap(self, _ax):
        if self.Dimension == '2D':
            cmap = mpl.cm.gnuplot
            cmap.set_bad('white',1.)
            if (self.ScattersPos is not None) or (isinstance(self.ScattersPos, Iterable) and self.ScattersPos.all is not None):
                self.placeScatters(_ax)
            theColorPlot = _ax.imshow(self.FrqMap, origin='lower', 
                                     extent=[self.Domain[0], self.Domain[-1], self.Domain[0],self.Domain[-1]],
                                     interpolation=None, cmap=cmap)
            plt.colorbar(theColorPlot, ax=_ax)
            _ax.set_title('Frequency [GHz]')
            _ax.set_ylabel("y [mm]")
            _ax.set_xlabel("x [mm]")
        elif self.Dimension == '1D':
            if (self.ScattersPos is not None) or (isinstance(self.ScattersPos, Iterable) and self.ScattersPos.all is not None):
                self.placeBarriers(_ax)
            _ax.plot(self.Domain, self.FrqMap)
            _ax.set_xlim(self.Domain[0], self.Domain[-1])
            _ax.set_ylabel('Frequency [GHz]')
        elif self.Dimension == 'Discrete_1D':
            self.placeBarriers(_ax)
            _ax.plot(self.ScattersPos+self.MatSize/2, self.FrqMap)
            _ax.set_xlim(np.min(self.ScattersPos)-self.MatSize, np.max(self.ScattersPos)+self.MatSize)
            _ax.set_ylabel('Frequency [GHz]')
        elif self.Dimension == 'Discrete_2D':
            cmap = mpl.cm.gnuplot
            themin = np.nanmin(self.FrqMap)
            themax = np.nanmax(self.FrqMap)
            _amps = (self.FrqMap-themin)/(themax-themin)
            for _i,_cyl in enumerate(self.ScattersPos):
                _circle = plt.Circle((_cyl), self.MatSize, color=cmap(_amps[_i]), lw=0)
                _ax[0].add_artist(_circle)
            cb = mpl.colorbar.ColorbarBase(_ax[1], cmap = cmap, ticks=[0, 0.25, 0.5, 0.75, 1])
            cb.ax.set_yticklabels(['%0.3f'%(themin),'%0.3f'%((themax+3*themin)*0.25),'%0.3f'%((themax+themin)*0.5),
                                   '%0.3f'%((3*themax+themin)*0.25),'%0.3f'%(themax)])
            _ax[0].set_title('Frequency [GHz]')
            _ax[0].set_ylabel("y [mm]")
            _ax[0].set_xlabel("x [mm]")
            _ax[0].set_xlim(np.min(self.ScattersPos[:,0])-self.MatSize-5, np.max(self.ScattersPos[:,0])+self.MatSize+5)
            _ax[0].set_ylim(np.min(self.ScattersPos[:,1])-self.MatSize-5, np.max(self.ScattersPos[:,1])+self.MatSize+5)
            
            
    def plotWidthMap(self, _ax):
        if self.Dimension == '2D':
            cmap = mpl.cm.gnuplot
            cmap.set_bad('white',1.)
            if (self.ScattersPos is not None) or (isinstance(self.ScattersPos, Iterable) and self.ScattersPos.all is not None):
                self.placeScatters(_ax)
            theColorPlot = _ax.imshow(self.WidthMap, origin='lower', 
                                     extent=[self.Domain[0], self.Domain[-1], self.Domain[0],self.Domain[-1]],
                                     interpolation=None, cmap=cmap)
            plt.colorbar(theColorPlot, ax=_ax)
            _ax.set_title('Width [GHz]')
            _ax.set_ylabel("y [mm]")
            _ax.set_xlabel("x [mm]")
        elif self.Dimension == '1D':
            if (self.ScattersPos is not None) or (isinstance(self.ScattersPos, Iterable) and self.ScattersPos.all is not None):
                self.placeBarriers(_ax)
            _ax.plot(self.Domain, self.WidthMap)
            _ax.set_xlim(self.Domain[0], self.Domain[-1])
            _ax.set_xlabel("x [mm]")
            _ax.set_ylabel('Width [GHz]')
        elif self.Dimension == 'Discrete_1D':
            self.placeBarriers(_ax)
            _ax.plot(self.ScattersPos+self.MatSize/2, self.WidthMap)
            _ax.set_xlim(np.min(self.ScattersPos)-self.MatSize, np.max(self.ScattersPos)+self.MatSize)
            _ax.set_xlabel("x [mm]")
            _ax.set_ylabel('Width [GHz]')
        elif self.Dimension == 'Discrete_2D':
            cmap = mpl.cm.gnuplot
            themin = np.nanmin(self.WidthMap)
            themax = np.nanmax(self.WidthMap)
            _amps = (self.WidthMap-themin)/(themax-themin)
            for _i,_cyl in enumerate(self.ScattersPos):
                _circle = plt.Circle((_cyl), self.MatSize, color=cmap(_amps[_i]), lw=0)
                _ax[0].add_artist(_circle)
            cb = mpl.colorbar.ColorbarBase(_ax[1], cmap = cmap, ticks=[0, 0.25, 0.5, 0.75, 1])
            cb.ax.set_yticklabels(['%0.3f'%(themin),'%0.3f'%((themax+3*themin)*0.25),'%0.3f'%((themax+themin)*0.5),
                                   '%0.3f'%((3*themax+themin)*0.25),'%0.3f'%(themax)])
            _ax[0].set_title('Width [GHz]')
            _ax[0].set_ylabel("y [mm]")
            _ax[0].set_xlabel("x [mm]")
            _ax[0].set_xlim(np.min(self.ScattersPos[:,0])-self.MatSize-5, np.max(self.ScattersPos[:,0])+self.MatSize+5)
            _ax[0].set_ylim(np.min(self.ScattersPos[:,1])-self.MatSize-5, np.max(self.ScattersPos[:,1])+self.MatSize+5)

            
    def plotIntHist(self, _ax):
        Int = np.abs(self.modePrime)**2/np.nanmean(np.abs(self.modePrime)**2)
        histInt, intBins, c = _ax.hist(Int, bins=np.linspace(0, np.percentile(Int, 99), 25), density=True, label=rf'$|\rho|={self.phaseRigidity:0.3f}$')
        _ax.set_yscale('log')
        _ax.set_xlabel(r"Intensity $I/\left\langleI\right\rangle$")
        _ax.set_ylabel(r'$p(I/\left\langleI\right\rangle)$')
        _ax.legend()
        
       
    def plotPhaseHist(self, _ax):
        ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        tickslabels = [r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"]
        histPhase, phaseBins, c=_ax.hist(np.angle(self.modePrime), bins=np.linspace(-np.pi, np.pi, 25), density=True, label=rf'$q^2={self.complexness:0.3f}$')
        _ax.set_xticks(ticks)
        _ax.set_xticklabels(tickslabels)
        _ax.set_xlabel(r'Phase')
        _ax.set_ylabel('$p($Phase$)$')
        _ax.legend()


    def plotFreqHist(self, _ax):
        theVMin = np.min([np.min(self.freq), np.mean(self.freq)-0.1])
        theVMax = np.max([np.max(self.freq), np.mean(self.freq)+0.1])
        bins = np.linspace(theVMin, theVMax, 25)
        histFreqs, freqsBins, c = _ax.hist(self.freq, bins=bins, density=True)
        _ax.set_xlabel('Frequency [GHz]')
        _ax.set_ylabel('$p($Frequency$)$')
        
        
    def plotWidthHist(self, _ax):
        theVMin = np.min([np.min(self.width), 0.5*np.mean(self.width)])
        theVMax = np.max([np.max(self.width), 1.5*np.mean(self.width)])
        bins = np.linspace(theVMin, theVMax, 25)
        histWidths, widthsBins, c = _ax.hist(self.width, bins=bins, density=True)
        _ax.set_xlabel('Width [GHz]')
        _ax.set_ylabel('$p($Width$)$')
        
        
    def placeBarriers(self, _ax):
        for _bar in self.ScattersPos:
            _ax.axvspan(_bar, _bar+self.MatSize, ls='--', alpha=0.2, color='k', lw=0.2)
            
            
    def placeScatters(self, _ax):
        for _cyl in self.ScattersPos:
            _circle = plt.Circle((_cyl), self.MatSize, color='k', alpha=0.1, ec=None)
            _ax.add_artist(_circle)