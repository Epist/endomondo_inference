from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers.core import Reshape
from keras.utils.data_utils import get_file
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

zMultiple = 5

data_path = "../multimodalDBM/endomondoHR_proper.json"
summaries_dir = "logs/keras/"
#endoFeatures = ["sport", "heart_rate", "gender", "altitude", "time_elapsed", "distance", "new_workout", "derived_speed", "userId"]
endoFeatures = ["sport", "heart_rate", "gender", "altitude", "time_elapsed", "distance", "new_workout", "derived_speed"]
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
model_file_name = "no_user_keras_test"

#pred_gen = endo_reader.endoIteratorSupervised(batch_size_m, num_steps, "test")
#pred_inputs, pred_targets = pred_gen.next()

for iteration in range(1, 20):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    base_size_limit = int(floor(num_samples/num_steps))

    #trainDataGen = endo_reader.generator_for_autotrain(batch_size_m, num_steps, "train", epoch_size_limit=int(base_size_limit*trainValTestSplit[0])) 
    trainDataGen = endo_reader.generator_for_autotrain(batch_size_m, num_steps, "train")  
    
    #history = model.fit_generator(trainDataGen, int(num_samples*trainValTestSplit[0]/batch_size_m), 1, validation_data=validDataGen, nb_val_samples = int(num_samples*trainValTestSplit[1]/batch_size_m))
    #model.fit_generator(trainDataGen, 2000, 1)
    
    #history = model.fit_generator(trainDataGen, int(floor(base_size_limit*trainValTestSplit[0]*num_steps)), 1)
    history = model.fit_generator(trainDataGen, base_size_limit*trainValTestSplit[0], 1)
    model.save(model_save_location+model_file_name+"_epoch_"+str(iteration))
    """try:
        with open(summaries_dir+"model_history_"+model_file_name+"_train_epoch_"+str(iteration), "wb") as f:
            pickle.dump(history, f)
        print("Model history saved")
    except:
        pass"""

    #validDataGen = endo_reader.generator_for_autotrain(batch_size_m, num_steps, "valid", epoch_size_limit=int(base_size_limit*trainValTestSplit[1]))
    validDataGen = endo_reader.generator_for_autotrain(batch_size_m, num_steps, "valid")
    #history = model.evaluate_generator(validDataGen, int(floor(base_size_limit*trainValTestSplit[1]*num_steps)))
    valid_score = model.evaluate_generator(validDataGen, base_size_limit*trainValTestSplit[1])
    print(valid_score)
    try:
        with open(summaries_dir+"model_score_"+model_file_name+"_valid_epoch_"+str(iteration), "wb") as f:
            pickle.dump(valid_score, f)
        print("Model score saved")
    except:
        pass

    
