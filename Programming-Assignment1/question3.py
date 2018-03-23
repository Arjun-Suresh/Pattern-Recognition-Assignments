import numpy
import matplotlib.pyplot as plt
import random
from math import *

sampleC0=[]
sampleC1=[]
covar = []
mean = []
covarEst = []
meanEstC0 = []
meanEstC1 = []
ldaErrors=[]
dldaErrors=[]
nmcErrors=[]
numSamples=[10,15,20,25,30,35,40]

def phi(x):
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

def createSample(num):
    n0 = numpy.random.binomial(num,0.5,1)[0]
    n1 = num - n0
    if n0 == 0 or n1 == 0:
        createSample(num)
    else:
        pointsC0 = numpy.random.multivariate_normal(mean[0], covar[0], n0).tolist()
        pointsC1 = numpy.random.multivariate_normal(mean[1], covar[0], n1).tolist()
        sampleC1.insert(0,pointsC1)
        sampleC0.insert(0,pointsC0)

def initValues(model, num):
    m1=[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
    m2=[[1,0.2,0,0,0,0],[0.2,1,0,0,0,0],[0,0,1,0.2,0,0],[0,0,0.2,1,0,0],[0,0,0,0,1,0.2],[0,0,0,0,0.2,1]]
    m3=[[1,0.2,0.2,0,0,0],[0.2,1,0.2,0,0,0],[0.2,0.2,1,0,0,0],[0,0,0,1,0.2,0.2],[0,0,0,0.2,1,0.2],[0,0,0,0.2,0.2,1]]
    if model == 0:
        covar.insert(0,numpy.matrix(m1))
    if model == 1:
        covar.insert(0,numpy.matrix(m2))
    if model == 2:
        covar.insert(0,numpy.matrix(m3))

    mean.insert(0,[0,0,0,0,0,0])
    mean.insert(1,[1,1,1,1,1,1])
    createSample(num)

def clearEstimatedValues():
    global covarEst
    global meanEstC0
    global meanEstC1
    covarEst = []
    meanEstC1 = []
    meanEstC0 = []

def clearAllValues():
    global sampleC0
    global sampleC1
    global covar
    global mean

    sampleC0=[]
    sampleC1=[]
    covar=[]
    mean=[]

    clearEstimatedValues()

def estimateValues(category):
    val=numpy.matrix([0,0,0,0,0,0]).T
    for i in range(len(sampleC0[0])):
            val = val + numpy.matrix(sampleC0[0][i]).T
    val = val * (1.0/len(sampleC0[0]))
    meanEstC0.insert(0, val)

    val=numpy.matrix([0,0,0,0,0,0]).T
    for i in range(len(sampleC1[0])):
            val = val + numpy.matrix(sampleC1[0][i]).T
    val = val * (1.0/len(sampleC1[0]))
    meanEstC1.insert(0, val)

    if category == 1:
        m1 = numpy.matrix([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
        m2 = m1
        for i in range(len(sampleC0[0])):
            matXC0 = numpy.matrix(sampleC0[0][i]).T
            meanMatC0 = meanEstC0[0]
            m1 = m1 + (matXC0 - meanMatC0) * (matXC0 - meanMatC0).T
        if len(sampleC0[0]) > 1:
            m1 = m1 * (1.0/(len(sampleC0[0])-1))

        for i in range(len(sampleC1[0])):
            matXC1 = numpy.matrix(sampleC1[0][i]).T
            meanMatC1 = meanEstC1[0]
            m2 = m2 + (matXC1 - meanMatC1) * (matXC1 - meanMatC1).T
        if len(sampleC1[0]) > 1:
            m2 = m2 * (1.0/(len(sampleC1[0])-1))
        m = (len(sampleC0[0])-1) * m1 + (len(sampleC1[0])-1) * m2
        m = m * (1.0/(len(sampleC0[0])+len(sampleC1[0])-2))

    elif category == 2:
        m=[]
        for k in range (6):
            val1 = 0.0
            val2 = 0.0
            for i in range(len(sampleC0[0])):
                val1 = val1 + (sampleC0[0][i][k] - meanEstC0[0][k].A[0][0]) * (sampleC0[0][i][k] - meanEstC0[0][k].A[0][0])

            for i in range(len(sampleC1[0])):
                val2 = val2 + (sampleC1[0][i][k] - meanEstC1[0][k].A[0][0]) * (sampleC1[0][i][k] - meanEstC1[0][k].A[0][0])

            val1 = val1/len(sampleC0[0])
            val2 = val2/len(sampleC1[0])
            val = (val1 * (len(sampleC0[0])-1) +val2 * (len(sampleC1[0])-1))/(len(sampleC0[0])+len(sampleC1[0])-2)
            arr=[]
            for j in range(6):
                if j == k:
                    arr.append(val)
                else:
                    arr.append(0)
            m.append(arr)
        m=numpy.matrix(m)

    else:
        m=numpy.matrix([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
    covarEst.insert(0,m)

def getError():
    probC0 = float(len(sampleC0[0])) / float((len(sampleC0[0]) + len(sampleC1[0])))
    probC1 = float(len(sampleC1[0])) / float((len(sampleC0[0]) + len(sampleC1[0])))
    p = log(float(len(sampleC1[0])) / float(len(sampleC0[0])))

    a = covarEst[0].I * (meanEstC1[0]-meanEstC0[0])
    b = -0.5 * ((meanEstC1[0]-meanEstC0[0]).T * covarEst[0].I * (meanEstC0[0]+meanEstC1[0])).A[0][0] + p
    val1 = ((a.T*numpy.matrix(mean[0]).T).A[0][0]+b)/(sqrt((a.T*covar[0]*a).A[0][0]))
    val2 = (-1.0 * (a.T*numpy.matrix(mean[1]).T).A[0][0]+b)/(sqrt((a.T*covar[0]*a).A[0][0]))
    err = probC0*phi(val1) + probC1*phi(val2)
    return err

def plot(model):
    plt.plot(numSamples, ldaErrors[model], label="LDA Model"+str(model))
    plt.plot(numSamples, dldaErrors[model], label="DLDA Model"+str(model))
    plt.plot(numSamples, nmcErrors[model], label="NMC Model"+str(model))
    plt.ylabel('error')
    plt.xlabel('number of samples')
    plt.title('Question 3')
    plt.legend()
    plt.show()

for model in range(3):
    ldaErr = []
    dldaErr = []
    nmcErr = []
    for num in numSamples:
        val1=0.0
        val2=0.0
        val3=0.0
        for run in range(100):
            clearAllValues()
            initValues(model, num)
            estimateValues(1)
            val1 = val1 + getError()
            clearEstimatedValues()
            estimateValues(2)
            val2 = val2 + getError()
            clearEstimatedValues()
            estimateValues(3)
            val3 = val3 + getError()
        val1 = val1/100.0
        val2 = val2/100.0
        val3 = val3/100.0
        ldaErr.append(val1)
        dldaErr.append(val2)
        nmcErr.append(val3)

    ldaErrors.insert(model, ldaErr)
    dldaErrors.insert(model,dldaErr)
    nmcErrors.insert(model, nmcErr)

    plot(model)


