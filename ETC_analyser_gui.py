#!/usr/bin/env python

# with the above statement the script can be called directly if made an executable

from __future__ import print_function
from __future__ import division

#! Python 3.6
"""
Open a file dialog window in tkinter using the filedialog method.
Tkinter has a prebuilt dialogwindow to access files. 
This example is designed to show how you might use a file dialog askopenfilename
and use it in a program.
"""
import os
import numpy as np
from numpy import mean,median,min,max,sqrt,std,abs,sum,percentile,log,log10,exp,floor,round
from scipy.special import erf, expit
# percentile(array,value in the range [0,100]
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from astropy.table import Table
import matplotlib
import platform
if(platform.system()=='Darwin'):
    matplotlib.use('MacOSX')
from matplotlib import style
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from scipy.optimize import newton
from extinction import *


class TkBase:

    def __init__(self, master):
        irow = 0 
        width=20
        self.scale = 1.0
        self.nrule = 0
        self.master = master
        self.ruleList =[]
        self.rulelmin = []
        self.rulelmax = []
        self.ruleNameList=[]
        self.ruleListText =[]
        self.skybright =['dark','grey','bright']
        self.spectrograph_arm=['blue','green','red']
        self.resolution=['lrs','hrs']
        self.logplot = IntVar()
        self.master.title( "Interactive Time Estimator v0.1")
        logo = PhotoImage(file = 'small_prism.png')
        label1 = ttk.Label(self.master,image=logo, borderwidth=0)
        label1.image = logo
        label1.grid(row = irow, column = 0, pady=2)
        ttk.Button(self.master, text = "Open file",
                   command = self.OpenFile, width=width).grid(row = irow, column = 1, pady=2)
        ttk.Button(self.master, text = "Exit",
                   command = lambda:exit(), width=width).grid(row = irow, column = 2, pady=2)

        irow += 1
        self.label2 = ttk.Label(self.master,width=80)
        self.label2.grid(row=irow,column=0,sticky='w',columnspan=3,pady=2,padx=5)
        irow += 1        
        #label2 = ttk.Label(self.master, text = "Signal-to-Noise",font=("Helvetica", 14))
        #label2.grid(row = irow, column = 0, columnspan = 3, pady=5)
        #irow += 1
        irow += 1
        self.skychoice = IntVar()
        #skyvalues=[0,2,4]
        skyvalues = [0,2,3]
        ttk.Radiobutton(self.master,text = 'Dark',value=0,variable=self.skychoice,
                        command=self.updateAllLabels).grid(row = irow, column = 0)
        ttk.Radiobutton(self.master,text = 'Grey',value=1,variable=self.skychoice,
                        command=self.updateAllLabels).grid(row = irow, column = 1)
        ttk.Radiobutton(self.master,text = 'Bright',value=2,variable=self.skychoice,
                        command=self.updateAllLabels).grid(row = irow, column = 2)
        irow += 1
        #
        self.armchoice = IntVar()
        ttk.Radiobutton(self.master,text = 'Blue',value=0,variable=self.armchoice,
                        command=self.updateAllLabels).grid(row = irow, column = 0)
        ttk.Radiobutton(self.master,text = 'Green',value=1,variable=self.armchoice,
                        command=self.updateAllLabels).grid(row = irow, column = 1)
        ttk.Radiobutton(self.master,text = 'Red',value=2,variable=self.armchoice,
                        command=self.updateAllLabels).grid(row = irow, column = 2)
        irow += 1
        #
        self.resolutionchoice = IntVar()
        ttk.Radiobutton(self.master,text = 'LRS',value=0,variable=self.resolutionchoice,
                        command=self.updateAllLabels).grid(row = irow, column = 0)
        ttk.Radiobutton(self.master,text = 'HRS',value=1,variable=self.resolutionchoice,
                        command=self.updateAllLabels).grid(row = irow, column = 1)
        irow += 1
        #
        self.MagLabel = ttk.Label(self.master,text='Ref. Mag:')
        self.MagLabel.grid(row=irow,column=0,sticky='e')
        self.Mag=ttk.Combobox(self.master, values=[16,17,18,19,20,21,22], width = width)
        self.Mag.grid(column=1,row=irow,sticky='w')
        self.Mag.bind("<Return>", self.scaleMag)
        self.Mag.bind("<<ComboboxSelected>>",self.scaleMag)
        ttk.Button(self.master, text = "Reset",
        command = self.reset, width=10).grid(row = irow, column = 2)
        irow += 1
        #
        self.DITLabel = ttk.Label(self.master,text='DIT(s):')
        self.DITLabel.grid(row=irow,column=0,sticky='e')
        self.DIT = ttk.Combobox(self.master, values=[60,600,1200,2400,3600,7200], width = width)
        self.DIT.bind("<Return>", self.printDIT)
        self.DIT.bind("<<ComboboxSelected>>",self.printDIT)
        self.DIT.grid(column=1,row=irow,sticky='w')
        
        #irow += 1
        #
        self.NDITLabel = ttk.Label(self.master,text='NDIT:')
        self.NDITLabel.grid(row=irow,column=1,sticky='e')
        self.NDIT = ttk.Combobox(self.master, values=[1,2,3,4], width = width)
        self.NDIT.bind("<Return>", self.printDIT)
        self.NDIT.bind("<<ComboboxSelected>>",self.printNDIT)
        self.NDIT.grid(column=2,row=irow,sticky='w')
        irow += 1
        #
        self.lminLabel = ttk.Label(self.master,text='lmin:')
        self.lminLabel.grid(row=irow,column=0,sticky='e')
        self.lmin = ttk.Combobox(self.master, values=[3900,4000,5300,5400,6100,7200], width = width)
        self.lmin.bind("<<ComboboxSelected>>",self.printlmin)
        self.lmin.bind("<Return>", self.printlmin)
        self.lmin.grid(row=irow,column=1,sticky='w')
        #irow += 1
        #
        self.lmaxLabel = ttk.Label(self.master,text='lmax:')
        self.lmaxLabel.grid(row=irow,column=1,sticky='e')
        self.lmax = ttk.Combobox(self.master, values=[4300,5200,5700,6800,7000,9400], width = width)
        self.lmax.bind("<<ComboboxSelected>>",self.printlmax)
        self.lmax.bind("<Return>", self.printlmax)
        self.lmax.grid(row=irow,column=2,sticky='w')
        irow += 1
        #
        self.skyvariationLabel = ttk.Label(self.master,text='Sky variation (%):')
        self.skyvariationLabel.grid(row=irow,column=0,sticky='e')
        self.skyvariation=ttk.Combobox(self.master, values=[0,5,10,15], width = width)
        self.skyvariation.insert(END, '0')
        self.skyvariation.bind("<Return>",self.printskyvariation)
        self.skyvariation.bind("<<ComboboxSelected>>",self.printskyvariation)
        self.skyvariation.grid(row=irow,column=1,sticky='w')
        #irow += 1
        #
        self.binsizeLabel = ttk.Label(self.master,text='Bin:')
        self.binsizeLabel.grid(row=irow,column=1,sticky='e')
        self.binsize=ttk.Combobox(self.master, values=[0,2,3,4,5], width = width)
        self.binsize.insert(END, '0')
        self.binsize.bind("<Return>", self.printbinsize)
        self.binsize.bind("<<ComboboxSelected>>",self.printbinsize)
        self.binsize.grid(row=irow,column=2,sticky='w')
        irow += 1
        self.ebvLabel = ttk.Label(self.master,text='E(B-V):')
        self.ebvLabel.grid(row=irow,column=0,sticky='e')
        self.ebv=ttk.Combobox(self.master, values=['0','0.01','0.05',
                                                   '0.1','0.2','0.5','1'], width = width)
        self.ebv.insert(END, '0')
        self.ebv.bind("<Return>", self.printebv)
        self.ebv.bind("<<ComboboxSelected>>",self.printebv)
        self.ebv.grid(row=irow,column=1,sticky='w')
        self.extmodel=ttk.Combobox(self.master, values=['Fitzpatrick 99',
                                                   'Fitzpatrick & Massa 07',
                                                   'Cardelli+ 89',
                                                   'O\' Donnell 94',
                                                    'Gordon+ 09'], width = width,state="readonly")
        self.extmodel.set('Fitzpatrick 99')
        #self.extmodel.insert(END, 'Fitzpatrick 99')
        self.extmodel.bind("<Return>", self.printebv)
        self.extmodel.bind("<<ComboboxSelected>>",self.printebv)
        self.extmodel.grid(row=irow,column=2,sticky='we')
        irow += 1
        # Button
        ttk.Button(self.master, text = "Plot SNR/pix vs AA",
                   command = self.PlotSNR, width=width).grid(row = irow, column = 0)
        ttk.Button(self.master, text = "Plot signal/pix vs AA",
                   command = self.PlotSignal, width=width).grid(row = irow, column = 1)
        ttk.Button(self.master, text = "Plot S & N",
                   command = self.PlotSignalNoise, width=width).grid(row = irow, column = 2)
        irow += 1
        ttk.Button(self.master, text = "Overplot SNR",
                   command = self.OverplotSNR, width=width).grid(row = irow, column = 0)
        ttk.Button(self.master, text = "Overplot signal",
                   command = self.OverplotSignal, width=width).grid(row = irow, column = 1)
        ttk.Button(self.master, text = "Overplot S & N",
                   command = self.OverplotSignalNoise, width=width).grid(row = irow, column = 2)
        irow += 1
        #ttk.Separator(self.master).grid(row = irow, column = 0,pady=2)
        #irow += 1
        ttk.Button(self.master, text = "Plot fluence+noise",
                   command = self.PlotObjNoise, width=width).grid(row = irow, column = 0)
        ttk.Button(self.master, text = "Plot obj+sky+noise",
                   command = self.PlotObjSkyNoise, width=width).grid(row = irow, column = 1)
        ttk.Button(self.master, text = "Plot sky+noise",
                   command = self.PlotSky, width=width).grid(row = irow, column = 2)
        irow += 1
        ttk.Button(self.master, text = "Overplot fluence+noise",
                   command = self.OverplotObjNoise, width=width).grid(row = irow, column = 0)  
        ttk.Button(self.master, text = "Overplot obj+sky+noise",
                   command = self.OverplotObjSkyNoise, width=width).grid(row = irow, column = 1)
        ttk.Button(self.master, text = "Overplot sky+noise",
                   command = self.OverplotSky, width=width).grid(row = irow, column = 2)
        irow += 1
        #
        ttk.Separator(self.master).grid(row = irow, column = 0,pady=5)
        #irow += 1
        #
        #self.plotWavelengthLabel = ttk.Label(self.master,text='Mouse (AA):')
        #self.plotWavelengthLabel.grid(row=irow,column=0,sticky='w')
        #ttk.Button(self.master, text = "Set to lmin",
        #           command = self.changelmin, width=width).grid(row = irow, column = 1)
        #ttk.Button(self.master, text = "Set to lmax",
        #           command = self.changelmax, width=width).grid(row = irow, column = 2)
        #irow += 1
        #
        #ttk.Separator(self.master).grid(row = irow, column = 0,pady=5)
        #irow += 1
        ttk.Button(self.master, text = "Mean SNR/pix",
                   command = self.meanSNR, width=width).grid(row = irow, column = 0,pady=2)
        ttk.Button(self.master, text = "Median SNR/pix",
                   command = self.medianSNR, width=width).grid(row = irow, column = 1,pady=2)
        ttk.Button(self.master, text = "Max SNR/pix",
                   command = self.maxSNR, width=width).grid(row = irow, column = 2,pady=2) 
        
        irow += 1
        ttk.Button(self.master, text = "Mean SNR/AA",
                   command = self.meanSNRAA, width=width).grid(row = irow, column = 0)
        ttk.Button(self.master, text = "Median SNR/AA",
                   command = self.medianSNRAA, width=width).grid(row = irow, column = 1)
        ttk.Button(self.master, text = "Max SNR/AA",
                   command = self.maxSNRAA, width=width).grid(row = irow, column = 2)
        irow += 1
        ttk.Button(self.master, text = "Emission Line SNR",
                   command = self.elineSNR, width=width).grid(row = irow, column = 0)
        ttk.Button(self.master, text = "Clear rule",
                   command = self.clearRule, width=width).grid(row = irow, column = 2)
        irow += 1
        self.successRuleLabel = ttk.Label(self.master,text='criterion:')
        self.successRuleLabel.grid(row=irow,column=0,sticky='e')
        self.successRule = ttk.Combobox(self.master, values=self.ruleList, width = width)
        self.successRule.bind("<<ComboboxSelected>>",self.evaluateCriterion)
        self.successRule.insert(END, 'mean(snr)')
        self.successRule.bind("<Return>", self.evaluateCriterion)
        self.successRule.grid(row=irow,column=1,sticky='w',columnspan=2)
        #
        self.crit = ttk.Label(self.master, text = "crtiterion", width=width,font = "Helvetica 11 italic")
        self.crit.grid(row=irow,column =2)
        irow += 1
        ttk.Label(self.master,text='Criterion name:').grid(row=irow, column=0,sticky='e')
        self.ruleName= ttk.Combobox(self.master, values=self.ruleNameList, width = width)
        self.ruleName.bind("<<ComboboxSelected>>",self.changeRule)
        self.ruleName.insert(END, 'Rule1')
        self.ruleName.grid(row=irow,column=1,sticky='w')
        ttk.Button(self.master, text="Add criterion to list", 
                   command=self.addRule).grid(row = irow, column = 2)
        irow += 1
        #ttk.Separator(self.master).grid(row = irow, column = 0,columnspan = 3,sticky='we')
        #irow += 1
        TexpLabel = ttk.Label(self.master,text='Criterion vs Texp',font = "Helvetica 12 bold").grid(row=irow,column=1,pady=5)
        irow += 1
        tminLabel = ttk.Label(self.master,text='min DIT (min):')
        tminLabel.grid(row=irow,column=0,sticky='e')
        self.tmin = Entry(self.master, width = width)
        self.tmin.insert(END, '1')
        self.tmin.bind("<Return>", self.printTmin)
        self.tmin.grid(row=irow,column=1,sticky='w')
        ttk.Button(self.master, text = "Plot",
                   command = self.PlottexCrit).grid(row = irow, column = 2)
        irow += 1
        self.tmaxLabel = ttk.Label(self.master,text='max DIT (max):')
        self.tmaxLabel.grid(row=irow,column=0,sticky='e')
        self.tmax = Entry(self.master, width = width)
        self.tmax.insert(END, '30')
        self.tmax.bind("<Return>", self.printTmax)
        self.tmax.grid(row=irow,column=1,sticky='w')
        ttk.Button(self.master, text = "Overplot",
                   command = self.OverplottexCrit).grid(row = irow, column = 2,pady=2)
        irow += 1
        # ---------
        #ttk.Separator(self.master).grid(row = irow, column = 0,columnspan = 3,sticky='we')
        #irow += 1
        self.pbrow = irow
        TexpLabel = ttk.Label(self.master,text='Texp vs Mag',font = "Helvetica 12 bold").grid(row=irow,column=1,pady=5)
        irow += 1
        self.approx=BooleanVar()
        self.approx.set('1') 
        ttk.Checkbutton(self.master,text='Approx binning',variable=self.approx).grid(row = irow, column = 2)
        maxtimeLabel = ttk.Label(self.master,text='Maximum DIT (min):')
        maxtimeLabel.grid(row=irow,column=0,sticky='e')
        self.maxtime = Entry(self.master, width = width)
        self.maxtime.insert(END, '30')
        self.maxtime.bind("<Return>", self.printMaxtime)
        self.maxtime.grid(row=irow,column=1,sticky='w')
        irow += 1
        RminLabel = ttk.Label(self.master,text='min Mag:')
        RminLabel.grid(row=irow,column=0,sticky='e')
        self.Rmin = Entry(self.master, width = width)
        self.Rmin.insert(END, '15')
        self.Rmin.bind("<Return>", self.printRmin)
        self.Rmin.grid(row=irow,column=1,sticky='w')
        self.log=BooleanVar()
        ttk.Checkbutton(self.master,text='ylog',variable=self.logplot).grid(row = irow, column = 2)
        irow += 1
        self.RmaxLabel = ttk.Label(self.master,text='max Mag:')
        self.RmaxLabel.grid(row=irow,column=0,sticky='e')
        self.Rmax = Entry(self.master, width = width)
        self.Rmax.insert(END, '22')
        self.Rmax.bind("<Return>", self.printRmax)
        self.Rmax.grid(row=irow,column=1,sticky='w')
        ttk.Button(self.master, text = "Plot",
                   command = self.PlottexpMag).grid(row = irow, column = 2)
        irow += 1
        #
        self.minMetricLabel = ttk.Label(self.master,text='min metric:')
        self.minMetricLabel.grid(row=irow,column=0,sticky='e')
        self.minMetric = Entry(self.master, width = width)
        self.minMetric.bind("<Return>", self.printMetric)
        self.minMetric.insert(END, '1')
        self.minMetric.grid(row=irow,column=1,sticky='w',pady=2)
        ttk.Button(self.master, text = "Overplot",
                   command = self.OverplottexpMag).grid(row = irow, column = 2,pady=2)
        #irow += 1
        #ttk.Button(self.master, text="Show catalogue manager",
        #                        command=self.showCatalogueWindow).grid(row = irow, column = 0,pady=2)
        #ttk.Button(self.master, text="Show rule manager",
        #                        command=self.showRuleWindow).grid(row = irow, column = 1,pady=2)
        irow += 1
        # ---------
        ttk.Separator(self.master).grid(row = irow, column = 0,columnspan = 3,sticky='we')
        irow += 1
        S = Scrollbar(self.master)
        S.grid(row=irow,column=0,sticky='w',pady=2,columnspan = 3)
        self.T = Text(self.master, height=1)
        self.T.grid(row=irow,column=0,sticky='w',pady=2,columnspan = 3)
        S.config(command=self.T.yview)
        self.T.config(yscrollcommand=S.set)
        #irow += 1
        
    def updateMetaData(self):
        self.meta = dict()
        self.meta['Path']= self.path
        self.meta['File']= self.name
        self.meta['RMag']= self.Mag.get()
        self.meta['DIT']= self.DIT.get()
        self.meta['NDIT']=self.NDIT.get()
        self.meta['lmin']= self.lmin.get()+'AA'
        self.meta['lmax']= self.lmax.get()+'AA'
        self.meta['Rule']= self.successRule.get()
        self.meta['SkyBrightness']= self.skybright[self.skychoice.get()]

    def printText(self,text):
        self.T.insert(END, text + '\n')
        self.T.see("end")
        
    def printNDIT(self,event):
        NDIT = float(self.NDIT.get())
        text = 'NDIT='+str(int(NDIT))
        self.printText(text)
        self.evaluateCriterion(None)
        self.updateLabel()
        
    def printDIT(self,event):
        time = self.DIT.get()
        try:
            ftime = float(time)
            if (ftime < 0):
                text='A miracle! Negative exposure time.'
                self.DIT.delete(0, END)
                self.DIT.insert(END,self.DITDefault)
            else:
                text='DIT:'+time+' s'
        except:
            text = 'Time should be a float! Set to the default value'
            self.DIT.delete(0, END)
            self.DIT.insert(END,self.DITDefault)
        self.printText(text)
        self.evaluateCriterion(None)
        self.updateLabel()
         
    def changelmin(self):
        self.lmin.delete(0, 'end')
        text = '{0:.3f}'.format(self.xmouse)
        self.lmin.insert(END,text)
        self.printlmin(None)

    def changelmax(self):
        self.lmax.delete(0, 'end')
        text = '{0:.3f}'.format(self.xmouse)
        self.lmax.insert(END,text)
        self.printlmax(None)

    def printlmin(self,event):
        lmin = self.lmin.get()
        try:
            flmin = float(lmin)
            if (flmin < 0):
                text='lmin is negative. Set to the default value'
                self.lmin.delete(0, END)
                self.lmin.insert(END,'{0:.3f}'.format(self.lminDefault))
            else:
                text='lmin:'+lmin+' AA' 
        except:
            text = 'lmin should be a float! Set to the default value'
            self.lmin.delete(0, END)
            self.lmin.insert(END,'{0:.3f}'.format(self.lminDefault))
        self.printText(text)
        self.evaluateCriterion(None)
        self.updateMetricLabel()

    def printlmax(self,event):
        lmax = self.lmax.get()
        try:
            flmax = float(lmax)
            if (flmax < 0):
                text='lmax is negative. Set to the default value'
                self.lmax.delete(0, END)
                self.lmax.insert(END,'{0:.3f}'.format(self.lmaxDefault))
            else:
                text='lmax:'+lmax+' AA' 
        except:
            text = 'lmax should be a float! Set to the default value'
            self.lmax.delete(0, END)
            self.lmax.insert(END,'{0:.3f}'.format(self.lmaxDefault))
        self.printText(text)
        self.evaluateCriterion(None)
        self.updateMetricLabel()

    def printbinsize(self,event):
        bin = self.binsize.get()
        try:
           ibin = int(bin)
           if(ibin<0):
               text = 'Negative bin size! Set to 0'
               self.binsize.delete(0, END)
               self.binsize.insert(END,0)
               bin='0'
           else:
               text='Bin size:'+bin+' pix'
        except:
            text = 'Bin size has to be an integer! Set to 0'
            self.binsize.delete(0, END)
            self.binsize.insert(END,0)
            bin='0'
        self.printText(text)
        self.evaluateCriterion(None)
        self.updateAllLabels()

    def printebv(self, event):
        ebv = self.ebv.get()
        try:
            febv = float(ebv)
            if(febv<0):
                text = 'Negative extinction! Set to 0.'
                self.ebv.delete(0, END)
                self.ebv.insert(END,0)
                ebv='0'
            else:
                 text='E(B-V):'+ebv
        except:
            text = 'Not a float for the extinction! Set to 0.'
            self.ebv.delete(0, END)
            self.ebv.insert(END,0)
            ebv='0'
        self.printText(text)
        text='Extinction model:' + self.extmodel.get()
        self.printText(text)
        x = self.tab['Wavelength'][self.find_where()]
        modelList={'Fitzpatrick 99':'f99',
                   'Fitzpatrick & Massa 07':'fm07',
                   'Cardelli+ 89':'ccm89',
                   'O\' Donnell 94':'od94',
                   'Gordon+ 09':'gcc09'}
        ebv = float(self.ebv.get())
        if(ebv > 0):
            extinctionMag = extinction(x, float(self.ebv.get()), 
                                       model=modelList[self.extmodel.get()])
            av = 10**(-0.4*extinctionMag)
            self.extinctionFlux = av
        else:
            self.extinctionFlux = np.full(x.size,1.)
        self.evaluateCriterion(None)
        self.updateAllLabels()
        
    def printskyvariation(self,event):
        skyvar = self.skyvariation.get()
        try:
           fskyvar = float(skyvar)
           if(fskyvar<0):
               text = 'Negative bin size! Set to 0'
               self.skyvariation.delete(0, END)
               self.skyvariation.insert(END,0)
               skyvar='0'
           else:
               text='Sky variation (%):'+skyvar
        except:
            text = 'Sky variation has to be a positive float! Set to 0'
            self.skyvariation.delete(0, END)
            self.skyvariation.insert(END,0)
            skyvar='0'
        self.printText(text)
        self.evaluateCriterion(None)
        self.updateAllLabels()

    def printTmin(self,event):
        default = '1'
        tmin = self.tmin.get()
        try:
            ftmin = float(tmin)
            if (ftmin < 0):
                text='A miracle! Negative exposure time. Set to the default value'
                self.printText(text)
                self.tmin.delete(0, END)
                self.tmin.insert(END,default)
        except:
            text = 'Time should be a float! Set to the default value'
            self.printText(text)
            self.tmin.delete(0, END)
            self.tmin.insert(END,default)
        text='min T:'+self.tmin.get()+' s'
        self.printText(text)

    def printTmax(self,event):
        default = '120'
        tmax = self.tmax.get()
        try:
            ftmax = float(tmax)
            if (ftmax < 0):
                text='A miracle! Negative exposure time. Set to the default value'
                self.printText(text)
                self.tmax.delete(0, END)
                self.tmax.insert(END,default)
        except:
            text = 'Time should be a float! Set to the default value'
            self.printText(text)
            self.tmax.delete(0, END)
            self.tmax.insert(END,default)        
        text='max T:'+self.tmax.get()+' s'
        self.printText(text)

    def printMaxtime(self,event):
        default = '2'
        maxtime = self.maxtime.get()
        try:
            fmaxtime = float(maxtime)
            if (fmaxtime < 0):
                text='A miracle! Negative exposure time. Set to the default value'
                self.printText(text)
                self.maxtime.delete(0, END)
                self.maxtime.insert(END,default)
        except:
            text = 'Time should be a float! Set to the default value'
            self.printText(text)
            self.maxtime.delete(0, END)
            self.maxtime.insert(END,default)
        text='maxtime (min):'+self.maxtime.get()
        self.printText(text)

    def printRmin(self,event):
        try:
            fmag = float(self.Rmin.get())
        except:
            text = 'Magnitude should be a float! Set to the default value'
            self.printText(text)
            self.Rmin.delete(0, END)
            self.Rmin.insert(END,'15')
        text='min Ref. Mag:'+self.Rmin.get()
        self.printText(text)
        
    def printRmax(self,event):
        try:
            fmag = float(self.Rmax.get())
        except:
            text = 'Magnitude should be a float! Set to the default value'
            self.printText(text)
            self.Rmax.delete(0, END)
            self.Rmax.insert(END,'22')
        text='max Ref. Mag:'+self.Rmax.get()
        self.printText(text)
        
    def printMetric(self,event):
        try:
            fmetric = float(self.minMetric.get())
        except:
            text = 'Metric should be a float! Set to the default value'
            self.printText(text)
            self.minMetric.delete(0, END)
            self.minMetric.insert(END,'1')
        text='metric: '+self.minMetric.get()
        self.printText(text)
        self.updateMetricLabel()
        
    def scaleMag(self,event):
        try:
            fmag = float(self.Mag.get())
        except:
            text = 'Magnitude should be a float! Set to the default value'
            self.printText(text)
            self.Mag.delete(0, END)
            self.Mag.insert(END,self.MagDefault)
        m0 = float(self.tab['Mag'][0])
        m1 = float(self.Mag.get())
        self.scale = 10**(0.4*(m0-m1))
        text='RMag= '+str(m1)+' Flux scaling factor:'+str(self.scale)
        self.printText(text)
        self.evaluateCriterion(None)
        self.updateLabel()
        self.updateRuleLabel()

    def evaluatePredefinedCriterion(self,rule):
        self.successRule.delete(0, 'end')
        self.successRule.insert(END, rule)
        self.evaluateCriterion(None)
        
    def computeCriterion(self):
        w = self.getWalengthRange()
        noise = self.noise[w]+abs(self.systematics[w])
        signal = self.signal[w]
        sky = self.sky[w]
        snr = signal/noise
        SNR = snr
        NOISE = noise
        SIGNAL = signal
        FLUENCE = signal
        fluence = signal
        flux = signal
        flux_density = signal
        aa = np.sqrt(self.pix_width[self.find_where()][w])
        ang = aa
        Ang = aa # replacing some C commands by Python
        input = self.successRule.get()
        input = input.lower()
        input = ''.join(input).replace("AND","and")\
                                   .replace("eq","==")\
                                   .replace("=","==")\
                                   .replace("gt",">")\
                                   .replace("lt","<")\
                                   .replace("ge",">=")\
                                   .replace("le","<=")\
                                   .replace("div","/")\
                                   .replace("mult","*")\
                                   .replace("OR","or")\
                                   .replace("&&","and ")\
                                   .replace("&","and ")\
                                   .replace("||","or")\
                                   .replace("^","**")\
                                   .replace("not"," not ")\
                                   .replace("pc"," percentile")
        self.input = ''.join(input).replace("====","==").replace(">==",">=").replace("<==","<=")
        self.printText("Converted rule: "+self.input)
        validWords = ['mean','median','min','max','sqrt',
                     'std','abs','sum','percentile',
                     'log','log10','exp','floor','round']        
        validDict = dict([ (k,np.__dict__.get(k)) for k in validWords])
        extraWords=['=','or','and','not','(',')','+','-','*','/','<','>','!',',',
                      'snr','noise','noise_cgs','sky','signal','flux','fluence','aa','ang',
                      '0','1','2','3','4','5','6','7','8','9','.']
        extraDict = dict([(k,k) for k in extraWords])
        extraDict = {k: k for k in extraWords}
        validDict.update(extraDict)
        validWords=validWords+extraWords
        string=''.join(self.input.split()) # remove whitespace
        for word in validWords:
            string=string.replace(word,'')
            l=len(string)
            if(l==0):
                break
        if(l!=0):
            print("Rule contains invalid symbol(s):")
            print(string)
            self.printText("Rule contains invalid symbol(s):")
            self.printText(string)
            valstr=', '.join(validWords)
            self.printText('List of major valid entries are:')
            self.printText(valstr)
            self.evalcrit=-999 # integer
            return
        try: # a="__import__('os').system('ls') security check"
             # b="_"+"_import_"+"_('os').system('ls')"
            self.evalcrit = eval(self.input)
            if(isinstance(self.evalcrit,bool)): # ok if the result is boolean
                return
            if(self.evalcrit.size > 1):
                self.printText('Output is an array. It should be a float. Return -999')
                self.evalcrit=-999
        except (IOError, AttributeError, SyntaxError, RuntimeError):
            print("Syntax error in rule.  Return -999")
            self.printText("Syntax error in rule.  Return -999")
            self.evalcrit=-999
        self.updateRuleLabel()
        self.updateMetricLabel()
        
    def evaluateCriterion(self,event):
        self.computeSignalNoise(float(self.DIT.get()),float(self.NDIT.get()))
        input = self.successRule.get()
        self.printText('rule: '+ input)
        self.computeCriterion()
        text = 'criterion: {0:.3f}'.format(self.evalcrit)
        self.printText(text)
        self.crit.configure(text=text)
        
    def meanSNR(self):
        rule = 'mean(snr)'
        self.evaluatePredefinedCriterion(rule)

    def meanSNRAA(self):
        rule = 'mean(snr/AA)'       
        self.evaluatePredefinedCriterion(rule)
        
    def medianSNR(self):
        rule = 'median(snr)'
        self.evaluatePredefinedCriterion(rule)
        
    def medianSNRAA(self):
        rule = 'median(snr/AA)'
        self.evaluatePredefinedCriterion(rule)
        
    def maxSNR(self):
        rule = 'max(snr)'
        self.evaluatePredefinedCriterion(rule)
        
    def maxSNRAA(self):
        rule = 'max(snr/AA)'
        self.evaluatePredefinedCriterion(rule)

    def elineSNR(self):
        rule = '(max(fluence)-median(fluence))/mean(noise)'
        self.evaluatePredefinedCriterion(rule)

    def clearRule(self):
        self.successRule.delete(0, 'end')
        self.crit.configure(text='criterion')
        
    def getWalengthRange(self):
        x = self.tab['Wavelength'][self.find_where()]
        lmin = float(self.lmin.get())
        lmax = float(self.lmax.get())
        if(lmin > lmax):
            messagebox.showinfo(title = 'Input error', message = 'Error lmin < lmax')
        else:
            w = (x >= lmin) & (x<=lmax)
            return w

    def updateRangeDefault(self):
        w = (self.tab['spectrograph_arm']==self.spectrograph_arm[self.armchoice.get()]) & (self.tab['spectrograph']==self.resolution[self.resolutionchoice.get()])
        self.lminDefault = self.tab['Wavelength'][w].astype(float).min()
        self.lmaxDefault= self.tab['Wavelength'][w].astype(float).max()
        self.lmin.delete(0, END)
        self.lmin.insert(END, '{0:.3f}'.format(self.lminDefault))
        self.lmax.delete(0, END)
        self.lmax.insert(END, '{0:.3f}'.format(self.lmaxDefault))
        
    def reset(self):
        self.Mag.delete(0, END)
        self.Mag.insert(END, self.MagDefault)
        self.updateRangeDefault()
        self.DIT.delete(0, END)
        self.DIT.insert(END,self.DITDefault)
        self.NDIT.delete(0, END)
        self.NDIT.insert(END,self.NDITDefault)
        self.skyvariation.delete(0, END)
        self.skyvariation.insert(END,0)
        self.binsize.delete(0, END)
        self.binsize.insert(END,0)
        self.ebv.delete(0, END)
        self.ebv.insert(END,0)
        self.extmodel.delete(0, END)
        self.extmodel.insert(END,'Fitzpatrick 99')
        self.printText("Resetting RMag, Texp, lmin, lmax, binsize, E(B-V) Fiztpatrick")
        self.printebv(None)
        self.label2.configure(text= self.name)
        self.evaluateCriterion(None)
        self.updateMetricLabel()
        return

    def OpenFile(self):
         self.cwd = os.getcwd()
         self.fullpath = askopenfilename(initialdir=self.cwd,
                                         filetypes =[("Fits File", "*.fits"),
                                                     ("All Files","*.*")],
                                         title = "Choose a ETC template fits file.")

         try:
            self.path, self.name = os.path.split(self.fullpath)
            self.printText("Openning file " + self.name)
            self.tab  = Table.read(self.fullpath,format='fits', hdu=1)
            self.MagDefault = self.tab['Mag'][0]
            self.DITDefault  = self.tab['DIT'][0]
            self.NDITDefault = int(self.tab['NDIT'][0])
            self.texp = self.tab['DIT'][0] * self.tab['NDIT'][0]
            fluence = self.tab['total_source_electrons']
            noise = self.tab['total_sky_electrons']
            fnan = (np.isnan(fluence) == False)
            fnoise = (np.isnan(noise) == False)
            if ((np.count_nonzero(fnan) > 0) or (np.count_nonzero(fnoise) > 0)):
                self.printText("NaNs in the input spectrum")
            wok = np.argwhere(fnan & fnoise & 
                              (self.tab['spectrograph_arm'] == 'blue'))
            self.lminDefault = self.tab['Wavelength'][wok].astype(float).min()
            self.lmaxDefault = self.tab['Wavelength'][wok].astype(float).max()
            self.dark = self.tab['Dark_current']
            self.readout = self.tab['SRON^2']
            self.pix_width = self.tab['pixel_width']
            self.printText("file read ok")
            self.fileok=True
            self.reset()
         except:
            messagebox.showinfo(title = 'File openning',
                                message = 'Error reading the file')

    def callback(self,event):
        print(event)
        #print(self.cbox.get())

    def callbackRmag(self,event):
        print(event)
        #print(self.Mag.get())

    def updateAllLabels(self):
        self.updateRangeDefault()
        self.updateLabel()
        self.updateRuleLabel()
        self.updateMetricLabel()
        self.evaluateCriterion(None)
        #self.printText('set to '+self.skybright[self.skychoice.get()])
        
    def updateLabel(self):
        self.textlabel = self.skybright[self.skychoice.get()]+' Mag='+self.Mag.get()+' DIT='+self.DIT.get()+'s'+' NDIT='+self.NDIT.get()+' Bin '+self.binsize.get()+' pix'+' E(B-V) '+self.ebv.get()

    def updateRuleLabel(self):
        self.rulelabel = self.skybright[self.skychoice.get()]+' Mag='+self.Mag.get()+' '+self.successRule.get()+' NDIT='+self.NDIT.get()+' Bin '+self.binsize.get()+' pix'+' E(B-V) '+self.ebv.get()
        
    def updateMetricLabel(self):
        self.metriclabel = self.skybright[self.skychoice.get()]+' '+self.successRule.get()+'='+self.minMetric.get()+' NDIT='+self.NDIT.get()+' Bin '+self.binsize.get()+' pix'+' E(B-V) '+self.ebv.get()
        
    def MakeplotSpectrum(self):
        self.second = Tk()
        self.second.title(self.title)
        frame = Frame(self.second)
        #matplotlib.use("TkAgg")
        #style.use('ggplot')
        canvas = FigureCanvasTkAgg(self.fig,master=self.second)
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        canvas.draw()
        canvas.callbacks.connect('button_press_event', self.on_click)
        irow = 0
        self.plotWavelengthLabel = ttk.Label(frame,text='Mouse (AA):')
        self.plotWavelengthLabel.grid(row=irow,column=0,sticky='w')
        ttk.Button(frame, text = "Set to lmin",
                   command = self.changelmin).grid(row=irow,column=1)
        ttk.Button(frame, text = "Set to lmax",
                   command = self.changelmax).grid(row=irow,column=2)
        ttk.Button(frame, text = "Close plot window",command=self.second.destroy).grid(row=irow,column=3)
        #creating toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.second )
        toolbar.update()
        frame.pack()

    def Makeplot(self):
        self.second = Tk()
        self.second.title(self.title)
        frame = Frame(self.second)
        canvas = FigureCanvasTkAgg(self.fig,master=self.second)
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        canvas.draw()
        canvas.callbacks.connect('button_press_event', self.on_click)
        irow = 0
        ttk.Button(frame, text = "Close plot window",command=self.second.destroy).pack()
        #creating toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.second )
        toolbar.update()
        frame.pack()

    def on_click(self,event):
        if event.inaxes is not None:
            ylim = self.p.get_ylim()
            self.xmouse = event.xdata
            lmin=float(self.lmin.get())
            lmax=float(self.lmax.get())
            text = 'Mouse (AA): {0:.3f}'.format(event.xdata)
            self.printText(text)
            #self.plotWavelengthLabel.configure(text=text)
        else:
            text = 'Clicked ouside axes bounds but inside plot window'
            self.printText(text)

    def closeWindow(self):
       self.second.destroy

    def binning(self,x):
        """
        Perform a running average
        """
        N=int(self.binsize.get())
        if (N == 0):
            return x
        else:
            xb = np.convolve(x, np.ones((N,))/N, mode='same')
            return xb

    def find_where(self):
        resolution    =  self.resolution[self.resolutionchoice.get()]
        skyBrightness =  self.skybright[self.skychoice.get()]
        spectrograph_arm = self.spectrograph_arm[self.armchoice.get()]
        #print(resolution,skyBrightness,spectrograph_arm)
        w = (self.tab['spectrograph']==resolution) & (self.tab['spectrograph_arm']==spectrograph_arm) \
            & (self.tab['Sky_brightness']==skyBrightness) & (self.tab['target_shape']=='point')
        if(np.count_nonzero(w)<=0):
            print('Error in the input file')
            sys.exit()
        return w

    def computeSignalNoise(self,DIT,NDIT,approximate=False):
        Nspec = 1
        Nspat = 5
        time = NDIT*DIT
        w = self.find_where()
        self.scaleMag = self.tab['Mag'][0]
        fluence = np.multiply(self.tab['total_source_electrons'][w],self.extinctionFlux)
        self.Wavelength = self.tab['Wavelength'][w]
        self.sky = self.tab['total_sky_electrons'][w]
        size = self.sky.shape[0]
        skyvariation = float(self.skyvariation.get())
        if(skyvariation>0):  
            self.systematics = self.sky*0.01*skyvariation*(1-2*np.random.rand(1,size)).reshape(size)
        else:
            self.systematics = np.zeros(size)
        self.signal = fluence*self.scale*(time/self.texp)
        
        # self.noise is the sigma of the total noise
        self.noise =np.sqrt(self.signal+self.sky*(time/self.texp)+Nspec*Nspat*self.dark[w][0]*time+NDIT*Nspat*self.readout[w][0])
        N=int(self.binsize.get())
        # check if one needs to rebin the spectra before computing the signal and noise
        if(N > 0):
            if (not approximate):
                size = self.noise.size
                nrealisations = 1000
                # normal(gaussian noise) with self.noise = sigma
                self.noise = std([self.binning(self.signal+rnd) for rnd in self.noise*np.random.randn(nrealisations,size)],axis=0)
                self.signal = self.binning(self.signal)
                self.sky = self.binning(self.sky)
            else:
                self.noise = self.noise/sqrt(float(N))
        else:
            return
              
    def PlotSNR(self):
        self.initiatePlot()
        self.OverplotSNR()

    def OverplotSNR(self):
        self.computeSignalNoise(float(self.DIT.get()),float(self.NDIT.get()))
        self.title='SNR'
        x = self.tab['Wavelength'][self.find_where()]
        if(float(self.skyvariation.get())>0):
            y = self.signal/(self.noise+abs(self.systematics))
        else:
            y = self.signal/self.noise
        self.p.plot(x,y,label=self.textlabel)
        self.p.set_xlabel('Wavelength (AA)')
        self.p.set_ylabel('no Units')
        self.p.legend()
        self.MakeplotSpectrum()

    def initiatePlot(self):
        self.fig = Figure(figsize=(7,5), dpi=100)
        self.p = self.fig.add_subplot(111)
        self.p.grid()
            
    def PlotSignal(self):
        self.initiatePlot()
        self.OverplotSignal()

    def PlotObjNoise(self):
        self.initiatePlot()
        self.OverplotObjNoise()

    def PlotObjSkyNoise(self):
        self.initiatePlot()
        self.OverplotObjSkyNoise()

    def OverplotObjSkyNoise(self):
        self.title='Fluence + Sky + Noise'
        self.computeSignalNoise(float(self.DIT.get()),float(self.NDIT.get()))
        size =  self.noise.shape[0]
        self.y = self.noise*np.random.randn(size)\
            +self.signal\
            +self.sky+self.systematics
        wneg=np.where(self.y < 0.)
        self.y[wneg] = 0.
        self.y = self.binning(self.y)
        self.OverplotRealisation()
        
    def OverplotObjNoise(self):
        self.title='Fluence + Noise'
        self.computeSignalNoise(float(self.DIT.get()),float(self.NDIT.get()))
        size =  self.noise.shape[0]
        self.y = self.noise*np.random.randn(size)+self.signal+self.systematics
        wneg=np.where(self.y < 0.)
        self.y[wneg] = 0.
        self.y = self.binning(self.y)
        self.OverplotRealisation()        
         
    def OverplotRealisation(self):
        x = self.tab['Wavelength'][self.find_where()]
        self.p.plot(x,self.y,label=self.textlabel)
        y = self.signal
        self.p.plot(x,y,label='Unbinned Fluence')
        if(int(self.binsize.get()) > 0):
            y = self.binning(y)
            self.p.plot(x,y,label='Fluence')
        self.p.set_xlabel('Wavelength (AA)')
        self.p.set_ylabel('ct / pix')
        self.p.legend(loc='best')
        self.MakeplotSpectrum()
    
    def PlotSignalNoise(self):
        self.initiatePlot()
        self.OverplotSignalNoise()
        
    def OverplotSignal(self):
        self.computeSignalNoise(float(self.DIT.get()),float(self.NDIT.get()))
        self.title='Fluence (Signal)'
        x = self.tab['Wavelength'][self.find_where()]
        y = self.signal
        self.p.plot(x,y,label=self.textlabel)
        self.p.set_xlabel('Wavelength (AA)')
        self.p.set_ylabel('ct / (pix s)')
        self.p.legend(loc='best')
        self.MakeplotSpectrum()
    
    def PlotSignalNoise(self):
        self.initiatePlot()
        self.OverplotSignalNoise()
        
    def OverplotSignalNoise(self):
        self.computeSignalNoise(float(self.DIT.get()),float(self.NDIT.get()))
        self.title='Flux and Noise'
        x = self.tab['Wavelength'][self.find_where()]
        y = self.signal
        self.p.plot(x,y,label='Signal '+self.textlabel)
        y = self.noise+self.systematics
        self.p.plot(x,y,label='Noise')
        self.p.set_xlabel('Wavelength (AA)')
        self.p.set_ylabel('ct / pix') # to change on he1srv
        self.p.legend(loc='best')
        self.MakeplotSpectrum()

    def PlotSky(self):
        self.initiatePlot()
        self.OverplotSky()

    def OverplotSky(self):
        self.computeSignalNoise(float(self.DIT.get()),float(self.NDIT.get()))
        self.title='Sky + Noise'
        x = self.tab['Wavelength'][self.find_where()]
        y = (self.sky+sqrt(self.sky)*np.random.randn(self.sky.size))
        y[np.where(y < 0)]=0.
        self.p.plot(x,y,label=self.skybright[self.skychoice.get()]+' Sky + sky noise')
        self.p.set_xlabel('Wavelength (AA)')
        self.p.set_ylabel('ct / pix') # to change on he1srv
        self.p.legend(loc='best')
        self.MakeplotSpectrum()
        
    def secant(self,tmin,tmax,mag,criterion,tol):
        a = tmin 
        b = tmax
        maxtime = 60*float(self.maxtime.get()) # limit it to min*60 = seconds
        input = self.successRule.get()
        #print("Criterion:",input) 
        while True:
            fb,flag = self.func(b,mag,criterion)
            if(flag==1):
                return 0
            if(fb > 0.):
                break
            b = b*10.
        maxiter = 10
        iter = 0
        while True:
            fa,flag = self.func(a,mag,criterion)
            if(flag==1):
                return 0
            fb,flag = self.func(b,mag,criterion)
            if(flag==1):
                return 0
            x = b - (fb * (b-a)) / (fb - fa)
            if (x<1.):
                x = 1
            if(x > maxtime):
                a = maxtime
                b = maxtime
                break
            a, b = b, x
            iter = iter + 1
            if (abs(a-b) < tol):
                break
            if(iter>maxiter):
                break
        return 0.5*(a+b)
            
    def func(self,t,mag,criterion):
        m0 = float(self.tab['Mag'][0])
        self.scale = 10**(0.4*(m0-mag)) # flux scaling with the magnitude
        NDIT = float(self.NDIT.get())
        self.computeSignalNoise(t,NDIT,approximate=self.approx.get()) # new signal and noise given t in seconds
        self.computeCriterion() # apply the success rule -> self.evalcrit
        if(self.evalcrit == -999):
            self.printText("Invalid criterion")
            flag = 1
            return 0,flag
        f = self.evalcrit-criterion
        flag = 0
        return f,flag

    def PlottexCrit(self):
        self.initiatePlot()
        self.OverplottexCrit()
                
    def OverplottexCrit(self):
        minute = 60.
        tmin=minute*float(self.tmin.get())
        tmax=minute*float(self.tmax.get())
        NDIT = float(self.NDIT.get())
        if(tmax <= tmin):
            text='tmax <= tmin!'
            self.printText(text)
            return
        dt = 30.
        crittime=[]
        time = np.arange(tmin,tmax,dt)
        for t in time:
            self.computeSignalNoise(t,NDIT,approximate=True)
            self.computeCriterion()
            if(self.evalcrit == -999):
                self.printText("Invalid criterion")
                return
            crittime.append(self.evalcrit)
        self.title='Criterion vs Texp'
        x = time*NDIT/60.
        y = crittime
        self.p.plot(x,y,label=self.rulelabel)
        self.p.set_title('Rule criterion vs time')
        self.p.set_xlabel('DITxNDIT (min.)')
        self.p.set_ylabel('Criterion value')
        self.p.legend(loc='best')
        self.Makeplot()
        
    def PlottexpMag(self):
        self.initiatePlot()
        self.OverplottexpMag()
        
    def OverplottexpMag(self):
        N= int(self.binsize.get())
        if (N>0 and (not self.approx.get())):
            dMag = 0.5
        else:
            dMag = 0.1
        tol = 1e-2
        minRMag = float(self.Rmin.get())
        maxRMag = float(self.Rmax.get())
        if(maxRMag <= minRMag):
            text='max R mag < min R mag'
            self.printText(text)
            return
        dt = sqrt(10**0.4*dMag)*2.
        magList = np.arange(minRMag,maxRMag,dMag)
        criterion = float(self.minMetric.get())
        #print('criterion:',criterion)
        texpMag=[]
        t = 1.0
        NDIT = float(self.NDIT.get())
        for mag in magList:
            t = self.secant(t,t+dt,mag,criterion,tol)
            if (t==0): # if the secant search fails, return 0
                return # there was an error in the rule
            texpMag.append(t*NDIT)
        self.title='Texp vs Mag'
        x = magList
        y = np.array(texpMag)/60.
        if (self.logplot.get()==1): # default value is 0
            self.p.semilogy(x,y,label=self.metriclabel)
        else:
            self.p.plot(x,y,label=self.metriclabel)
        self.p.set_title('Exposure time vs Magnitude')
        self.p.set_xlabel('Mag')
        self.p.set_ylabel('texp=DITxNDIT (min.)')
        self.p.legend(loc='best')
        self.Makeplot()

    def showCatalogueWindow(self,event=None):
        catalogueWindow(self)

    def showRuleWindow(self,event=None):
        self.r=ruleWindow(self)

    def updateRuleWindow(self,event=None):
        ruleWindow.updateRuleList(self)
        
    def changeRule(self,event=None):
        ind = self.ruleNameList.index(self.ruleName.get()) # get the index of the rule in the list
        self.lmin.delete(0, END)
        self.lmin.insert(END, '{0:.3f}'.format(self.rulelmin[ind]))
        self.lmax.delete(0, END)
        self.lmax.insert(END, '{0:.3f}'.format(self.rulelmax[ind]))
        self.successRule.delete(0,END) # delete and replace
        self.successRule.insert(END, self.ruleList[ind])
        self.evaluateCriterion(None)
        
    def addRule(self):
        self.nrule += 1
        #self.ruleList.append([self.nrule,self.ruleName.get(),self.lmin.get(),self.lmax.get(),self.successRule.get()])
        text=str(self.nrule)+' '+self.ruleName.get()+'  L_MIN ='+str(self.lmin.get())+'  L_MAX='+str(self.lmax.get())+'  rule='+self.successRule.get()
        self.ruleList.append(self.successRule.get())
        self.ruleNameList.append(self.ruleName.get())
        self.ruleListText.append(text)
        self.rulelmin.append(float(self.lmin.get()))
        self.rulelmax.append(float(self.lmax.get()))
        self.successRule.config(values=self.ruleList)
        self.ruleName.config(values=self.ruleNameList)
        #r = ruleWindow(self)
        #ruleWindow.updateRuleList(self.r,'bla')
        #self.updateRuleWindow(None)
        
    def setToRule(self):
        rule = ruleList[int(self.irule.split()[0])]
        self.lmin.delete(0, END)
        self.lmin.insert(END, '{0:.3f}'.format(rule[2]))
        self.lmax.delete(0, END)
        self.lmax.insert(END, '{0:.3f}'.format(rule[3]))
        self.printText("use parameters in list")
        self.successRule.insert(END,rule[4])
        self.evaluateCriterion(None)
        self.updateMetricLabel()
        return

class ruleWindow(Toplevel):

    def __init__(self,master):
        self.catalogue = master
        self.irule = 0
        self.ruleList=[]
        width = 20
        Toplevel.__init__(self)
        self.title( "Interactive Time Estimator v0.1")
        logo = PhotoImage(file = 'small_prism.png')
        label1 = ttk.Label(self,image=logo, borderwidth=0)
        label1.image = logo
        irow = 1
        label1.grid(row = irow, column = 0, pady=2)
        ttk.Button(self, text = "Close window",
                   command = lambda:self.destroy(),
                   width=width).grid(row = irow, column = 2, pady=2)
        irow += 1
        ttk.Separator(self).grid(row = irow, column = 0,pady=5)
        irow +=1
        ttk.Button(self, text = "Open rule file",
                   width=width).grid(row = irow, column = 1, pady=2)
        ttk.Button(self, text = "Save rule file",
                   width=width).grid(row = irow, column = 2, pady=2)
        headers=['Name','lmin','lmax','rule']
        nbHeaders = 4
        irow += 1
        ttk.Label(self,text='Ruleset:').grid(row=irow, column=0,sticky='e')
        ttk.Entry(self,width=4*width).grid(row=irow,column=1,sticky='w',columnspan=3,pady=5)
        irow += 1
        ttk.Label(self,text='List of Rules:').grid(row=irow, column=0,sticky='e')
        irow += 1
        S = Scrollbar(self)
        S.grid(row=irow,column=0,sticky='w',pady=2,columnspan = 5)
        self.T = Text(self, height=10)
        self.T.grid(row=irow,column=0,sticky='w',pady=2,columnspan = 5)
        S.config(command=self.T.yview)
        self.T.config(yscrollcommand=S.set)
        if len(self.catalogue.ruleListText) > 0:
            for text in self.catalogue.ruleListText:
                self.T.insert(END, text + '\n')
        self.T.see("end")

        def updateRuleList(self,text):
            self.T.insert(END, text + '\n')
        
class catalogueWindow(Toplevel):

    def __init__(self,master):
        self.catalogue = master
        self.irule = 0
        self.ruleList=[]
        width = 20
        Toplevel.__init__(self)
        self.title( "Interactive Time Estimator v0.1")
        logo = PhotoImage(file = 'small_prism.png')
        label1 = ttk.Label(self,image=logo, borderwidth=0)
        label1.image = logo
        irow = 1
        label1.grid(row = irow, column = 0, pady=2)
        ttk.Button(self, text = "Open catalogue file",
                   width=width).grid(row = irow, column = 1, pady=2)
        ttk.Button(self, text = "Exit",
                   command = lambda:exit(),
                   width=width).grid(row = irow, column = 2, pady=2)
        irow += 1
        ttk.Label(self,text='Sub catalogue ID:').grid(row=irow, column=1,sticky='e')
        self.subcatID=ttk.Entry(self,width=width).grid(row=irow,column=2,sticky='w')
        irow += 1
        ttk.Separator(self).grid(row = irow, column = 0,pady=5)
        irow +=1
        ttk.Button(self, text = "Open rule file",
                   width=width).grid(row = irow, column = 1, pady=2)
        ttk.Button(self, text = "Save rule file",
                   width=width).grid(row = irow, column = 2, pady=2)
        headers=['Name','lmin','lmax','rule']
        nbHeaders = 4
        irow += 1
        ttk.Label(self,text='Ruleset:').grid(row=irow, column=0,sticky='e')
        ttk.Entry(self,width=4*width).grid(row=irow,column=1,sticky='w',columnspan=3,pady=5)
        irow += 1
        ttk.Label(self,text='max texp (min):').grid(row=irow, column=0,sticky='e')
        ttk.Entry(self,width=width).grid(row=irow,column=1,sticky='w')
        irow +=1
        ttk.Label(self,text='histogram bin size:').grid(row=irow, column=0,sticky='e')
        ttk.Entry(self,width=width).grid(row=irow,column=1,sticky='w')
        ttk.Button(self, text = "Plot Magnitude histogram",
                   width=width).grid(row = irow, column = 2)
        irow += 1
        ttk.Separator(self).grid(row = irow, column = 0,pady=5)
        irow += 1
        ttk.Label(self,text='min Ref. Mag:').grid(row=irow, column=0,sticky='e')
        ttk.Entry(self,width=width).grid(row=irow,column=1,sticky='w')
        ttk.Label(self,text='max Ref. Mag:').grid(row=irow, column=2,sticky='e')
        ttk.Entry(self,width=width).grid(row=irow,column=3,sticky='w')
        irow += 1
        ttk.Button(self, text = "Apply ruleset",
                   width=width).grid(row = irow, column = 2, pady=5)
        irow += 1
        ttk.Label(self,text='Rule name:').grid(row=irow, column=0,sticky='e')
        self.ruleName=ttk.Entry(self,width=width)
        self.ruleName.insert(END, 'NoName')
        self.ruleName.grid(row=irow,column=1,sticky='w')
        ttk.Button(self, text="Adopt rule", 
                   command=self.adoptRule).grid(row = irow, column = 2)
        irow += 1
        ttk.Label(self,text='Rules').grid(row=irow, column=1,sticky='we')
        self.irow = irow + 1
        
    def addRule(self):
        self.irule += 1
        cat = self.catalogue
        text=str(self.irule)+' '+self.ruleName.get()+' lmin ='+str(cat.lmin.get())+' lmax='+str(cat.lmax.get())+' rule='+cat.successRule.get()
        self.ruleList.append(text)
        ttk.Combobox(self, values=self.ruleList, width = 60).grid(row=self.irow,column=0,columnspan=3)

def main():
    master = Tk()
    app = TkBase(master)
    master.geometry("800x1200")
    master.mainloop()

if __name__ == "__main__": main()

# TOTAL = FLUENCE + SKY
# NOISE = SQRT(FLUENCE+SKY + DARK + READOUT)
# DARK = 3. (e-/hr/pix)
# READOUT 2.3, 2.5 and 3.0 e-/pix

# DARK = 3/3660..*t
# texp = tab['TEXP']
