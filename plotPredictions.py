%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np

def loadData(fileName):
    print("Loading data file")
    with open(fileName, "rb") as f:
        data = pickle.load(f)
    #except:
    #    raise(IOError("Data file: " + fileName + " not in valid pickle format"))
    print("Data file loaded")
    return data
        
            
class dataEpoch(object):
    def __init__(self, targetSeq, logitSeq, epochNum):
        self.targetSeq = targetSeq
        self.logitSeq = logitSeq
        self.epochNum = epochNum
        self.endoFeatures = endoFeatures
        self.targetAtt = targetAtt
        self.modelType = model
        self.lossType = lossType
        self.trainValTestSplit = trainValTestSplit
        self.modelRunIdentifier = modelRunIdentifier
        
def transformData(data, sequenceLength=500, scalingFactor=1):
    predictionsList = data.logitSeq
    targetsList = data.targetSeq
    
    numRows = int(len(predictionsList)/500)
    predictionsByWorkout = np.zeros((numRows, sequenceLength))
    targetsByWorkout = np.zeros((numRows, sequenceLength))
    #Now transform the lists into n x 500 arrays
    for i in range(numRows):
        predictionsByWorkout[i,:] = predictionsList[i*sequenceLength : (i+1)*sequenceLength]
        targetsByWorkout[i,:] = targetsList[i*sequenceLength : (i+1)*sequenceLength]
    
    return (predictionsByWorkout*scalingFactor, targetsByWorkout*scalingFactor)

def plotPerformance(data, plotTitle=None, plotErrorBars=False):
    predictionsByWorkout, targetsByWorkout = transformData(data, scalingFactor=250)
    residuals = predictionsByWorkout-targetsByWorkout
    #Compute the means and standard deviations of the residuals
    means = residuals.mean(0)
    stds = residuals.std(0)
    indices=np.array(range(len(means)))
    #Plot the mean exercise sequence with error bars representing the standard deviation
    fig = plt.figure()
    if plotErrorBars:
        plt.errorbar(indices, means, yerr=stds)
    else:
        plt.plot(indices, means)
        
    #Label the plot with the epoch number, the model type, the error measure, and the data variables
    if plotTitle is None:
        mt = data.modelType
        lt = data.lossType
        ef = str(data.endoFeatures)
        en = str(data.epochNum)
        plt.title(mt + ", " + lt + ", " + ef + ", epoch:" + en)
    else:
        plt.title(plotTitle)
    MAE=np.mean(abs(means))
    fig.suptitle("Mean absolute error: " + str(MAE))
    plt.show()
