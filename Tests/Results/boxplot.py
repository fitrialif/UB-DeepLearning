import matplotlib.pyplot as plt
import numpy as np
import csv

threeLayerFM = []
threeLayerNDFM = []
fourLayer02FM = []
fourLayer04FM = []
fourLayer1632 = []
fourLayer3264 = []
fourLayerBN = []
fourLayerDA = []
DAv = []

def fileReader():
    with open("3-32-043-32-043-32-043-32-043-32-043-32-043-32-043-32-043-32-043-32-04_UB_Calcium_PreRec.csv") as filename:
        template=csv.reader(filename)
        for row in template:
            countC = 0
            recall = 0
            prec = 0
            for column in row:
                if(countC==0):
                    recall = float(column)
                if(countC==1):
                    prec = float(column)
                countC+=1
            fmeasure = 2*((recall*prec)/(recall+prec))
            threeLayerFM.append(fmeasure)
    with open("3-BN-16-043-BN-16-043-BN-16-043-BN-16-043-BN-16-043-BN-16-043-BN-16-043-BN-16-043-BN-16-043-BN-16-04_UB_Calcium_PreRec.csv") as filename:
        template=csv.reader(filename)
        for row in template:
            countC = 0
            recall = 0
            prec = 0
            for column in row:
                if(countC==0):
                    recall = float(column)
                if(countC==1):
                    prec = float(column)
                countC+=1
            fmeasure = 2*((recall*prec)/(recall+prec))
            threeLayerNDFM.append(fmeasure)
    with open("3-BN-32-013-BN-32-013-BN-32-013-BN-32-013-BN-32-013-BN-32-013-BN-32-013-BN-32-013-BN-32-013-BN-32-01_UB_Calcium_PreRec.csv") as filename:
        template=csv.reader(filename)
        for row in template:
            countC = 0
            recall = 0
            prec = 0
            for column in row:
                if(countC==0):
                    recall = float(column)
                if(countC==1):
                    prec = float(column)
                countC+=1
            fmeasure = 2*((recall*prec)/(recall+prec))
            fourLayer02FM.append(fmeasure)
    with open("3-BN-32-043-BN-32-043-BN-32-043-BN-32-043-BN-32-043-BN-32-043-BN-32-043-BN-32-043-BN-32-043-BN-32-04_UB_Calcium_PreRec.csv") as filename:
        template=csv.reader(filename)
        for row in template:
            countC = 0
            recall = 0
            prec = 0
            for column in row:
                if(countC==0):
                    recall = float(column)
                if(countC==1):
                    prec = float(column)
                countC+=1
            fmeasure = 2*((recall*prec)/(recall+prec))
            fourLayer04FM.append(fmeasure)

# fake up some data
print('______________________RRRRRRRRR_________________________')
fileReader()
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)
labels = (['3L 32 04','3L BN 16 04', '3L BN 32 01', '3L BN 32 04'])
# basic plot
plt.boxplot([threeLayerFM,threeLayerNDFM,fourLayer02FM,fourLayer04FM])
plt.xticks(np.arange(len(labels))+1,labels)

plt.show()
