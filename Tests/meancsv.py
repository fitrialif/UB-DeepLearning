import csv
import re
from matplotlib import pyplot as PLT
from matplotlib import cm as CM
from matplotlib import mlab as ML
import numpy as NP
import math
import numpy as np
import seaborn as sns;
import seaborn as sns; sns.set()

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
	x = X.ravel()
	y = Y.ravel()
	z = ZD.ravel()
	gridsize=30
	PLT.subplot(111)

	PLT.hexbin(XD, YD, C=allData, gridsize=6, cmap=CM.jet, bins=None)
	PLT.axis([1, 5, -1, 7])

	cb = PLT.colorbar()
	cb.set_label('Net Accuracy')

	PLT.savefig("results.png")

def heatmapGenerator():
	all_data = np.asarray(allData)
	all_data = all_data.reshape((7,3))
	plot  = sns.heatmap(all_data)
	plot.invert_yaxis()
	fig = plot.get_figure()
	fig.savefig("test.png")

def fileReader():
	count = 1
	countRow = 0
	num = []
	with open("results4Layers.csv") as filename:
	    template=csv.reader(filename)
	    for row in template:
	    	rowData = []
	        for column in row:
	        	if(countRow==2 or countRow==3):
	        		sumCount=0
	        		sumA=0
	        		num = re.findall("\d+\.\d+", column)
	        		for floats in num:
					print(floats)
        				sumA += float(floats)
	        			sumCount+=1
				sumA = sumA / sumCount
				rowData.append(sumA)
			countRow+=1
	    	count+=1
	    	countRow=0
	    	allData.append(rowData)
		with open("results3LayerDrop_Form_Beta.csv","w+") as my_csv:
		    csvWriter = csv.writer(my_csv,delimiter=',')
		    csvWriter.writerows(allData)
fileReader()
print(allData)
