import numpy
import matplotlib.pyplot as plt
from math import *

sampleC0=[]
sampleC1=[]
covar = []
mean = []
covarEst = []
meanEstC0 = []
meanEstC1 = []
xDLDAEnd = []
yDLDAEnd = []
xLDAEnd = []
yLDAEnd = []
xNMCEnd = []
yNMCEnd = []

def phi(x):
    return (1.0 + erf(x / sqrt(2.0))) / 2.0
def createSample(model):
    sampleC0.insert(model,numpy.random.multivariate_normal(mean[0], covar[model], 5))
    sampleC1.insert(model,numpy.random.multivariate_normal(mean[1], covar[model], 5))

def initValues():
    covar.insert(0,numpy.matrix([[1,0],[0,1]]))
    covar.insert(1,numpy.matrix([[1,0.2],[0.2,1]]))
    mean.insert(0,[0,0])
    mean.insert(1,[1,1])
    createSample(0)
    createSample(1)

def clearValues():
    for i in range(len(covarEst)):
        j = covarEst.pop()
    for i in range(len(meanEstC0)):
        j = meanEstC0.pop()
        j = meanEstC1.pop()

def estimateValues(model,category):
    val1 = 0.0
    val2 = 0.0
    for i in range(5):
        val1 = val1 + sampleC0[model][i][0]
        val2 = val2 + sampleC0[model][i][1]
    val1 = val1 / 5.0
    val2 = val2 / 5.0
    meanModelClassEst = [val1, val2]
    meanEstC0.insert(model, numpy.matrix(meanModelClassEst).T)

    val1 = 0.0
    val2 = 0.0
    for i in range(5):
        val1 = val1 + sampleC1[model][i][0]
        val2 = val2 + sampleC1[model][i][1]
    val1 = val1 / 5.0
    val2 = val2 / 5.0
    meanModelClassEst = [val1, val2]
    meanEstC1.insert(model, numpy.matrix(meanModelClassEst).T)
    if category == 1:
        m1 = numpy.matrix([[0,0],[0,0]])
        m2 = numpy.matrix([[0,0],[0,0]])
        for i in range(5):
            matXC0 = numpy.matrix(sampleC0[model][i]).T
            matXC1 = numpy.matrix(sampleC1[model][i]).T
            meanMatC0 = meanEstC0[model]
            meanMatC1 = meanEstC1[model]
            m1 = m1 + (matXC0 - meanMatC0) * (matXC0 - meanMatC0).T
            m2 = m2 + (matXC1 - meanMatC1) * (matXC1 - meanMatC1).T
        m1 = m1 * 0.2
        m2 = m2 * 0.2
        m = 4 * m1 + 4 * m2
        m = m * (1.0/8.0)

    else:
        m=[]
        for k in range (2):
            val1 = 0.0
            val2 = 0.0
            for i in range(5):
                val1 = val1 + (sampleC0[model][i][k] - meanEstC0[model][k].A[0][0]) * (sampleC0[model][i][k] - meanEstC0[model][k].A[0][0])
                val2 = val2 + (sampleC1[model][i][k] - meanEstC1[model][k].A[0][0]) * (sampleC1[model][i][k] - meanEstC1[model][k].A[0][0])
            val1 = val1/5.0
            val2 = val2/5.0
            val = (val1+val2)/2
            arr=[]
            for j in range(2):
                if j == k:
                    arr.append(val)
                else:
                    arr.append(0)
            m.append(arr)
        m=numpy.matrix(m)
    covarEst.insert(model,m)

def getDecisionBoundary(model,category):
    maxY = sampleC0[model][0][1]
    for i in range(5):
        if maxY < sampleC0[model][i][1]:
            maxY = sampleC0[model][i][1]
    for i in range(5):
        if maxY < sampleC1[model][i][1]:
            maxY = sampleC1[model][i][1]

    minY = sampleC0[model][0][1]
    for i in range(5):
        if minY > sampleC0[model][i][1]:
            minY = sampleC0[model][i][1]
    for i in range(5):
        if minY > sampleC1[model][i][1]:
            minY = sampleC1[model][i][1]

    if category < 3:
        b = 0.5 * (meanEstC1[model] - meanEstC0[model]).T * covarEst[model].I * (meanEstC1[model] + meanEstC0[model])
        b = b.A[0][0]
        coeff = (covarEst[model].I * (meanEstC1[model] - meanEstC0[model])).T
        xVal1 = (b-coeff.A[0][1]*minY) / coeff.A[0][0]
        xVal2 = (b-coeff.A[0][1]*maxY) / coeff.A[0][0]
    else:
        xVal1 = maxY*(meanEstC0[model][1].A[0][0]-meanEstC1[model][1].A[0][0])/(meanEstC1[model][0].A[0][0]-meanEstC0[model][0].A[0][0])
        xVal2 = minY*(meanEstC0[model][1].A[0][0]-meanEstC1[model][1].A[0][0])/(meanEstC1[model][0].A[0][0]-meanEstC0[model][0].A[0][0])

    xPoints = [xVal1, xVal2]
    yPoints = [minY, maxY]
    if category == 1:
        xLDAEnd.insert(model, xPoints)
        yLDAEnd.insert(model, yPoints)
    elif category == 2:
        xDLDAEnd.insert(model, xPoints)
        yDLDAEnd.insert(model, yPoints)
    else:
        xNMCEnd.insert(model, xPoints)
        yNMCEnd.insert(model, yPoints)

def getError(model):
    probC0 = float(len(sampleC0[0])) / float((len(sampleC0[0]) + len(sampleC1[0])))
    probC1 = float(len(sampleC1[0])) / float((len(sampleC0[0]) + len(sampleC1[0])))
    p = log(float(len(sampleC1[0])) / float(len(sampleC0[0])))

    a = covarEst[model].I * (meanEstC1[model] - meanEstC0[model])
    b = -0.5 * ((meanEstC1[model] - meanEstC0[model]).T * covarEst[model].I * (meanEstC0[model] + meanEstC1[model])).A[0][0] + p
    val1 = ((a.T * numpy.matrix(mean[0]).T).A[0][0] + b) / (sqrt((a.T * covar[model] * a).A[0][0]))
    val2 = (-1.0 * (a.T * numpy.matrix(mean[1]).T).A[0][0] + b) / (sqrt((a.T * covar[model] * a).A[0][0]))
    err = probC0 * phi(val1) + probC1 * phi(val2)
    return err

def plot(model):
    xSampleC0=[]
    ySampleC0=[]
    xSampleC1=[]
    ySampleC1=[]

    for i in range(5):
        xSampleC0.append(sampleC0[model][i][0])
    for i in range(5):
        xSampleC1.append(sampleC1[model][i][0])
    for i in range(5):
        ySampleC0.append(sampleC0[model][i][1])
    for i in range(5):
        ySampleC1.append(sampleC1[model][i][1])
    plt.scatter(xSampleC0, ySampleC0, label="class 0", color="green",
                marker="o", s=30)
    plt.scatter(xSampleC1, ySampleC1, label="class 1", color="red",
                marker="X", s=30)
    plt.plot(xLDAEnd[model], yLDAEnd[model], label="LDA Model"+str(model))
    plt.plot(xDLDAEnd[model], yDLDAEnd[model], label="DLDA Model"+str(model))
    plt.plot(xNMCEnd[model], yNMCEnd[model], label="NMC Model"+str(model))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Question 2')
    plt.legend()
    plt.show()

initValues()
estimateValues(0,1)
getDecisionBoundary(0,1)
LDAError0 = getError(0)
estimateValues(1,1)
getDecisionBoundary(1,1)
LDAError1 = getError(1)

clearValues()
estimateValues(0,2)
getDecisionBoundary(0,2)
DLDAError0 = getError(0)
estimateValues(1,2)
getDecisionBoundary(1,2)
DLDAError1 = getError(1)

clearValues()
estimateValues(0,3)
getDecisionBoundary(0,3)
NMCError0 = getError(0)
estimateValues(1,3)
getDecisionBoundary(1,3)
NMCError1 = getError(1)

plot(0)
plot(1)
print ("Model1 LDA error: "+str(LDAError0))
print ("Model2 LDA error: "+str(LDAError1))
print ("Model1 DLDA error: "+str(DLDAError0))
print ("Model2 DLDA error: "+str(DLDAError1))
print ("Model1 NMC error: "+str(NMCError0))
print ("Model2 NMC error: "+str(NMCError1))