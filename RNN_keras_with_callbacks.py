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
import sys

from data_interpreter_Keras import dataInterpreter, metaDataEndomondo
from inputManager import inputManager
import pickle
from math import floor
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import datetime
from parse_args_keras import parse_args_keras

#parse_args_keras(sys.argv)

patience = 3
max_epochs = 50

zMultiple = 5

data_path = "../multimodalDBM/endomondoHR_proper.json"
summaries_dir = "logs/keras/"
#endoFeatures = ["sport", "heart_rate", "gender", "altitude", "time_elapsed", "distance", "new_workout", "derived_speed", "userId"]
endoFeatures = ["heart_rate", "gender", "altitude", "time_elapsed", "distance", "new_workout", "derived_speed"]
trainValTestSplit = [0.8, 0.1, 0.1]
targetAtt = "heart_rate"
inputOrderNames = [x for x in endoFeatures if x!=targetAtt]
trimmed_workout_len = 450
num_steps = 128
batch_size_m = 64


endo_reader = dataInterpreter(fn=data_path, scaleVals=True, trimmed_workout_length=trimmed_workout_len)
endo_reader.buildDataSchema(endoFeatures, targetAtt, trainValTestSplit, zMultiple)
input_dim = endo_reader.getInputDim(targetAtt)
target_dim = endo_reader.getTargetDim(targetAtt)

#num_samples = int((trimmed_workout_len*endo_reader.numDataPoints))
num_samples = 81274880

print('Build model...')
model = Sequential()
#model.add(Reshape((batch_size_m, num_steps, input_dim), batch_input_shape=(batch_size_m*num_steps, input_dim)))
model.add(LSTM(128, return_sequences=True, batch_input_shape=(batch_size_m, num_steps, input_dim), stateful=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, stateful=True))
model.add(Dropout(0.2))
model.add(Dense(target_dim))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer='rmsprop')
print("Endomodel Built!")


model_save_location = "/home/lmuhlste/endomondo_inference/model_states/"
model_file_name = "keras_fixedZscores_patience3_noUser_noSport"

#pred_gen = endo_reader.endoIteratorSupervised(batch_size_m, num_steps, "test")
#pred_inputs, pred_targets = pred_gen.next()

modelRunIdentifier = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
model_file_name += modelRunIdentifier #Applend a unique identifier to the filenames

best_model = None
best_valid_score = 9999999999
best_epoch = 0
for iteration in range(1, max_epochs):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    base_size_limit = int(floor(num_samples/num_steps))

    #trainDataGen = endo_reader.generator_for_autotrain(batch_size_m, num_steps, "train", epoch_size_limit=int(base_size_limit*trainValTestSplit[0])) 
    trainDataGen = endo_reader.generator_for_autotrain(batch_size_m, num_steps, "train")  
    
    #history = model.fit_generator(trainDataGen, int(num_samples*trainValTestSplit[0]/batch_size_m), 1, validation_data=validDataGen, nb_val_samples = int(num_samples*trainValTestSplit[1]/batch_size_m))
    #model.fit_generator(trainDataGen, 2000, 1)
    
    #history = model.fit_generator(trainDataGen, int(floor(base_size_limit*trainValTestSplit[0]*num_steps)), 1)


    model_save_fn = model_save_location+model_file_name+"_epoch_"+str(iteration)
    checkpoint = ModelCheckpoint(model_save_fn, verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')


    #history = model.fit_generator(trainDataGen, base_size_limit*trainValTestSplit[0], 1, callbacks=[checkpoint, early_stopping], validation_data=validDataGen,
    #    nb_val_samples=base_size_limit*trainValTestSplit[1])
    #history = model.fit_generator(trainDataGen, base_size_limit*trainValTestSplit[0], 1, callbacks=[checkpoint, early_stopping])
    history = model.fit_generator(trainDataGen, base_size_limit*trainValTestSplit[0], 1, callbacks=[checkpoint])

    try:
        del history.model
        with open(summaries_dir+"model_history_"+model_file_name+"_epoch_"+str(iteration), "wb") as f:
            pickle.dump(history, f)
        print("Model history saved")
    except:
        pass

    validDataGen = endo_reader.generator_for_autotrain(batch_size_m, num_steps, "valid")
    #validDataGen = endo_reader.generator_for_autotrain(batch_size_m, num_steps, "valid", epoch_size_limit=int(base_size_limit*trainValTestSplit[1]))
    #valid_score = model.evaluate_generator(validDataGen, int(floor(base_size_limit*trainValTestSplit[1]*num_steps)))
    valid_score = model.evaluate_generator(validDataGen, base_size_limit*trainValTestSplit[1])
    print(valid_score)
    try:
        with open(summaries_dir+"model_valid_score_"+model_file_name+"_epoch_"+str(iteration), "wb") as f:
            pickle.dump(valid_score, f)
        print("Model validation score saved")
    except:
        pass

    if valid_score <= best_valid_score:
        best_valid_score = valid_score
        #best_model = 
        best_epoch = iteration
    elif (iteration-best_epoch <= patience):
        pass
    else:
        print("Stopped early at epoch: " + str(iteration))
        break


best_model = keras.models.load_model(model_save_location+model_file_name+"_epoch_"+str(best_epoch))
best_model.save(model_save_location+model_file_name+"_bestValidScore")

print("Best epoch: " + str(best_epoch) + "   validation score: " + str(best_valid_score))

print("Testing")
testDataGen = endo_reader.generator_for_autotrain(batch_size_m, num_steps, "test")
test_score = model.evaluate_generator(testDataGen, base_size_limit*trainValTestSplit[2])
print("Test score: " + str(test_score))


test_set = endo_reader.testSet

with open(summaries_dir+model_file_name+"_testSet.p", "wb") as f:
    pickle.dump(test_set, f)
    print("Model test set saved")

print("Done!!!")
