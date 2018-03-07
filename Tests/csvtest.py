import csv
import re
from matplotlib import pyplot as PLT
from matplotlib import cm as CM
from matplotlib import mlab as ML
import numpy as NP
import math
allData = []
XD = []
YD = []
def graphGenerator():
	n = 1e5
	x = y = NP.linspace(-5, 5, 100)
	X, Y = NP.meshgrid(x, y)
	Z1 = ML.bivariate_normal(X, Y, 2, 2, 0, 0)
	Z2 = ML.bivariate_normal(X, Y, 4, 1, 1, 1)
	ZD = Z2 - Z1
	print(ZD)
	x = X.ravel()
	y = Y.ravel()
	z = ZD.ravel()
	gridsize=30
	PLT.subplot(111)

	PLT.hexbin(XD, YD, C=allData, gridsize=gridsize, cmap=CM.jet, bins=None)
	PLT.axis([1, 6, 1, 6])

	cb = PLT.colorbar()
	cb.set_label('mean value')
	PLT.show() 

def fileReader():
	count = 1
	countRow = 0
	num = []
	with open("resultsAllFin.csv") as filename:
	    template=csv.reader(filename)
	    for row in template:
	    	rowData = []
	        for column in row:
	        	if(countRow==4):
	        		sumCount=0
	        		sumA=0
	        		num = re.findall("\d+\.\d+", column)
	        		for floats in num:
	        			if(sumCount==99):
	        				rowData.append(float(floats))
	        			sumCount+=1
	        	else:
	        		if(countRow==0):
	        			YD.append(math.log(int(column),2))
        			if(countRow==1):
        				XD.append(math.log(int(column),2))
	        	countRow+=1
	    	count+=1
	    	countRow=0
	    	allData.append(rowData)
fileReader()
graphGenerator()
print(allData)
print(XD)
print(YD)