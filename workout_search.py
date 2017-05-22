#For recommending workouts with a desired target sequence attribute

#Compute the AUC

#User chooses a workout that they want to find a similar heart rate profile for
#Choose x workouts at random
#Compute the "distances" between the heart rates generated by the model for each of these x workouts and the target heart rate sequence
#Rank the x workouts in order of best fit to target (lowest distance)
#Find the rank of the original workout in this list
#Return the list of workout numbers and their corresponding similarity scores in order


from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers.core import Reshape
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import random
import sys, argparse

#from data_interpreter_Keras import dataInterpreter, metaDataEndomondo
from data_interpreter_Keras_multiTarget import dataInterpreter, metaDataEndomondo
from inputManager import inputManager
import pickle
from math import floor
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import datetime
from parse_args_keras import parse_args_keras
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import random
from fastdtw import fastdtw
from scipy.interpolate import interp1d
import sklearn
from sklearn import linear_model
from sklearn.externals import joblib

#Params
target_workout_index = 191470 #93082 #1234 #Choose a workout to use as the target heart rate!!!
number_to_search = 9999
target_variable = "heart_rate" #Can only use a single target variable
distance_metric = "DTW"
modelFN = "model_states/keras__all_hrTarget02_58PM_April_11_2017_bestValidScore" 



data_path = "../multimodalDBM/endomondoHR_proper_newmeta_copy.json"
trainValTestFN = "logs/keras/keras__noSport_hrTarget" #The filename root from which to load the train valid test split

endoFeatures = ["heart_rate", "new_workout", "gender", "sport", "userId", "altitude", "distance", "derived_speed", "time_elapsed"]
targetAtts = ["heart_rate"]
trainValTestSplit = [0.8, 0.1, 0.1]
trimmed_workout_len = 450
scale_toggle = False
scaleTargets= False #"scaleVals" #False
zMultiple = 5
samples = 5000 #For resampling 
resample_toggle = False
model_type = "LSTM" # "LSTM" "linear"

endoReader = dataInterpreter(fn=data_path, scaleVals=scale_toggle, trimmed_workout_length=trimmed_workout_len, scaleTargets=scaleTargets)
endoReader.buildDataSchema(endoFeatures, targetAtts, trainValTestSplit = trainValTestSplit, zMultiple = zMultiple, trainValidTestFN = trainValTestFN)

if target_workout_index not in endoReader.dataPointList:
	raise(exception("The target workout is in the excised list!!! Choose another..."))

print("Target workout index:", target_workout_index)

class workout(object):
	def __init__(self, workoutIndex):
		self.workoutIndex = workoutIndex
		self.inputSeq = None
		self.targetSeq = None
		#self.rawInputSeq = None
		#self.rawTargetSeq = None
		self.timeElapsed = None
		self.loadData()
		self.predictedTrace = None
		self.evalScore = None

	def loadData(self):
		#Load the inputs and targets from the json file and scale them appropriately
		#Target scaling toggling is taken care of by the object paramaters for endoreader, not the function call to getDpByIndex()
		(x, y) = endoReader.getDpByIndex(self.workoutIndex, scaling=scale_toggle)
		self.inputSeq = x
		self.targetSeq = np.array(y).astype(np.float)
 
def getInputByName(input_data, varName, endoReader):
	#Provides access to decoded input variable sequences by attribute name
	dataKey = endoReader.decoderKey()
	dataIndex = dataKey.index(varName)
	batch_size = np.shape(input_data)[0]
	varData = []
	for i in range(batch_size):
	    decodedData = np.array(endoReader.dataDecoder(input_data[i]))
	    varData.extend(decodedData[:, dataIndex])
	return varData

def predict_model(model, inputSeq, model_type="LSTM"):
	if model_type == "LSTM":
		return model.predict_on_batch(inputSeq)
	elif model_type == "linear":
		preds = np.zeros([1,trimmed_workout_len,1])
		for i, timepoint in enumerate(inputSeq[0]):
			#print(timepoint)
			#print(i)
			preds[0,i,0] = model.predict(timepoint)
		return preds

def eval_model(predictedTrace, targetSeq, distance_metric, samples = None, prediction_timesteps = None, target_timesteps = None):
	#return model.test_on_batch(targetSeq)
	if distance_metric == "MSE":
		return mean_squared_error(predictedTrace, targetSeq)
	if distance_metric == "MAE":
		return mean_absolute_error(predictedTrace, targetSeq)
	if distance_metric == "DTW":
		#Resampling and dynamic time warping
		if resample_toggle:
			predictedTrace= resample(predictedTrace, prediction_timesteps, samples)
			targetSeq = resample(targetSeq, target_timesteps, samples)

		#distance, path = fastdtw(predictedTrace.astype(np.float), targetSeq.astype(np.float))
		distance, path = fastdtw(predictedTrace, targetSeq)
		return distance

def resample(values, timestamps, samples, interp_kind = 'linear'):
    #interpolate and resample sequence with constant intervals
    #print("resampling")
    f = interp1d(timestamps, values, kind=interp_kind)

    min_time = timestamps[0]
    max_time = timestamps[len(timestamps)-1]

    times = [(max_time-min_time)*(x/samples)+min_time for x in range(samples)]
    resampled = [f(x) for x in times]
    
    return resampled

def rescaleZscoredData(endo_reader, sequence, att, zMultiple):
    #Removes the z score scaling. Does this by getting the varmeans and stds from the endo_reader, 
    #and then performing arithmetic operations on the data sequence
        
    variableMeans = endo_reader.variableMeans
    variableStds = endo_reader.variableStds
    
    unMult = [x/float(zMultiple) for x in sequence]
    diff = [x*float(variableStds[att]) for x in unMult]
    raw = [x+float(variableMeans[att]) for x in diff]
    return raw


availableWorkouts = [x for x in endoReader.dataPointList if x != target_workout_index]

def genRandomWorkoutList(numWorkouts, availableWorkouts):
	#From the available workouts, select numWorkouts, generate the workout objects, and add them to a list
	random.shuffle(availableWorkouts)
	wo_indices = availableWorkouts[0:numWorkouts]
	return [workout(x) for x in wo_indices]

def workoutFitCompare(wo1, wo2):
	if wo1.evalScore>wo2.evalScore:
		return -1 #The second workout has lower error and therefore should come first
	elif wo1.evalScore<wo2.evalScore:
		return 1 #The first workout has lower error and therefore should come first
	else:
		return 0

#Generate the list of random workouts
workoutList = genRandomWorkoutList(number_to_search, availableWorkouts)

#Add the target workout to the list of random workouts
targetWorkout = workout(target_workout_index)
workoutList.append(targetWorkout)

#Load a trained model
def load_and_rebuild_model(modelFN, num_steps, input_dim, target_dim):
	if model_type == "LSTM":

		oldModel = keras.models.load_model(modelFN)#Load a model that has already been trained

		print('Build model...')
		model = Sequential()
		#model.add(Reshape((batch_size_m, num_steps, input_dim), batch_input_shape=(batch_size_m*num_steps, input_dim)))
		model.add(LSTM(128, return_sequences=True, batch_input_shape=(1, num_steps, input_dim), stateful=True))
		model.add(Dropout(0.2))
		model.add(LSTM(128, return_sequences=True, stateful=True))
		model.add(Dropout(0.2))
		model.add(Dense(target_dim))
		model.add(Activation('linear'))

		model.compile(loss='mean_squared_error', optimizer='rmsprop')

		model.set_weights(oldModel.get_weights())#Transfer the weights from the old model to the new model
		print("Endomodel Built!")

		return model
	elif model_type == "linear":
		#with open(modelFN, "rb") as f:
		#	models = pickle.load(models, f)
		#	model = models[0]
		model = joblib.load(modelFN)
    	return model

model = load_and_rebuild_model(modelFN, trimmed_workout_len, endoReader.getInputDim(targetAtts), endoReader.getTargetDim(targetAtts))

#Compute the predicted heart rate sequences on each workout in the list
print("Predicting workouts")
for i, wo in enumerate(workoutList):
	if i%100==0:
		print("Predicted " + str(i) + " workouts so far")
	inputSeq = wo.inputSeq
	targetSeq = targetWorkout.targetSeq
	predictedTrace = predict_model(model, inputSeq, model_type)

	target_times = targetWorkout.timeElapsed
	prediction_timesteps = wo.timeElapsed
	#Make sure to grab the unscaled version. Do this using code from endreader, which should be able to get the raw sequence straight from the datapoint

	#Need to rescale both the target and the prediction if they were computed using zscore-scaling
	predictedTrace = predictedTrace[0,:,:].flatten()
	targetSeq = targetSeq[0,:,:].flatten()
	predictedTrace = predictedTrace.astype(float)

	if scaleTargets == True:
		#Unscale the targets
		predictedTrace = rescaleZscoredData(endoReader, predictedTrace, targetAtts[0], zMultiple)
		targetSeq = rescaleZscoredData(endoReader, targetSeq, targetAtts[0], zMultiple)

	#Compute the similarity for each heart rate sequence (Could do this by using the target HR seq as the target seq in the model)
	eval_score = eval_model(predictedTrace, targetSeq, distance_metric, samples = samples, prediction_timesteps = prediction_timesteps, target_timesteps = target_times)
	wo.predictedTrace = predictedTrace
	wo.evalScore = eval_score

#Sort the list of workouts by similarityScore
#Impliment a custom comparator for workout objects. Then use a library quicksort to sort the list
print("Sorting workouts")
#workoutList.sort(workoutFitCompare)
workoutList.sort(key=lambda x: x.evalScore)

#Find the position of the targetWorkout in the sorted list of workouts
def findTargetRank(sortedWOList, target_workout_index):
	for i, wo in enumerate(sortedWOList):
		if wo.workoutIndex == target_workout_index:
			return i+1 #Rank is indexed from 1
	raise(exception("Target workout not in the list"))

print("Target workout index:", target_workout_index)
targetRank = findTargetRank(workoutList, target_workout_index)
print("Target rank: " + str(targetRank) + " out of " + str(number_to_search+1))
print("Target " + distance_metric + " is " + str(workoutList[targetRank-1].evalScore))
print("Best " + distance_metric + " is " + str(workoutList[0].evalScore))
print("Worst " + distance_metric + " is " + str(workoutList[number_to_search].evalScore))
print("The AUC is " + str(1-(targetRank/number_to_search)))

print("The indices of the top 10 workouts are: ", [wo.workoutIndex for wo in workoutList[0:10]])

#Save the ordered list of workouts as well as the targetWorkout and the paramaters 


#Provide tools for visualizing the matches

