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


class keras_endoLSTM(object):
    def __init__(self, cmdArgs):

        #self.model_save_location = "/home/lmuhlste/endomondo_inference/model_states/"
        self.model_save_location = "./model_states/"
        self.model_file_name = "keras_fixedZscores_patience3_noUser_noSport"
        self.patience = 3
        self.max_epochs = 50

        self.zMultiple = 5

        self.data_path = "../multimodalDBM/endomondoHR_proper.json"
        self.summaries_dir = "logs/keras/"

        #endoFeatures = ["sport", "heart_rate", "gender", "altitude", "time_elapsed", "distance", "new_workout", "derived_speed", "userId"]
        self.endoFeatures = ["heart_rate", "gender", "altitude", "time_elapsed", "distance", "new_workout", "derived_speed"]
        self.trainValTestSplit = [0.8, 0.1, 0.1]
        self.targetAtts = ["heart_rate"]
        self.inputOrderNames = [x for x in self.endoFeatures if x not in self.targetAtts]
        self.trimmed_workout_len = 450
        self.num_steps = 128
        self.batch_size_m = 64

        self.scale_toggle = True #Should the data values be scaled to their z-scores with the z-multiple?

        parse_args_keras(cmdArgs, self)

        self.endo_reader = dataInterpreter(fn=self.data_path, scaleVals=self.scale_toggle, trimmed_workout_length=self.trimmed_workout_len)
        self.endo_reader.buildDataSchema(self.endoFeatures, self.targetAtts, self.trainValTestSplit, self.zMultiple)
        self.input_dim = self.endo_reader.getInputDim(self.targetAtts)
        self.target_dim = self.endo_reader.getTargetDim(self.targetAtts)

        #num_samples = int((trimmed_workout_len*endo_reader.numDataPoints))
        self.num_samples = 81274880

        self.model = self.build_model()

    def build_model(self):

        print('Build model...')
        model = Sequential()
        #model.add(Reshape((batch_size_m, num_steps, input_dim), batch_input_shape=(batch_size_m*num_steps, input_dim)))
        model.add(LSTM(128, return_sequences=True, batch_input_shape=(self.batch_size_m, self.num_steps, self.input_dim), stateful=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))
        model.add(Dense(self.target_dim))
        model.add(Activation('linear'))

        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        print("Endomodel Built!")

        return model


    def run_model(self, model):


        #pred_gen = endo_reader.endoIteratorSupervised(batch_size_m, num_steps, "test")
        #pred_inputs, pred_targets = pred_gen.next()

        modelRunIdentifier = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
        self.model_file_name += modelRunIdentifier #Applend a unique identifier to the filenames

        #best_model = None
        best_valid_score = 9999999999
        best_epoch = 0

        base_size_limit = int(floor(self.num_samples/self.num_steps))

        for iteration in range(1, self.max_epochs):
            print()
            print('-' * 50)
            print('Iteration', iteration)

            #trainDataGen = endo_reader.generator_for_autotrain(batch_size_m, num_steps, "train", epoch_size_limit=int(base_size_limit*trainValTestSplit[0])) 
            trainDataGen = self.endo_reader.generator_for_autotrain(self.batch_size_m, self.num_steps, "train")  
            
            #history = model.fit_generator(trainDataGen, int(num_samples*trainValTestSplit[0]/batch_size_m), 1, validation_data=validDataGen, nb_val_samples = int(num_samples*trainValTestSplit[1]/batch_size_m))
            #model.fit_generator(trainDataGen, 2000, 1)
            
            #history = model.fit_generator(trainDataGen, int(floor(base_size_limit*trainValTestSplit[0]*num_steps)), 1)


            model_save_fn = self.model_save_location+self.model_file_name+"_epoch_"+str(iteration)
            checkpoint = ModelCheckpoint(model_save_fn, verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')


            #history = model.fit_generator(trainDataGen, base_size_limit*trainValTestSplit[0], 1, callbacks=[checkpoint, early_stopping], validation_data=validDataGen,
            #    nb_val_samples=base_size_limit*trainValTestSplit[1])
            #history = model.fit_generator(trainDataGen, base_size_limit*trainValTestSplit[0], 1, callbacks=[checkpoint, early_stopping])
            history = model.fit_generator(trainDataGen, base_size_limit*self.trainValTestSplit[0], 1, callbacks=[checkpoint])

            try:
                del history.model
                with open(self.summaries_dir+"model_history_"+self.model_file_name+"_epoch_"+str(iteration), "wb") as f:
                    pickle.dump(history, f)
                print("Model history saved")
            except:
                pass

            validDataGen = self.endo_reader.generator_for_autotrain(self.batch_size_m, self.num_steps, "valid")
            #validDataGen = endo_reader.generator_for_autotrain(batch_size_m, num_steps, "valid", epoch_size_limit=int(base_size_limit*trainValTestSplit[1]))
            #valid_score = model.evaluate_generator(validDataGen, int(floor(base_size_limit*trainValTestSplit[1]*num_steps)))
            valid_score = model.evaluate_generator(validDataGen, base_size_limit*self.trainValTestSplit[1])
            print(valid_score)
            try:
                with open(self.summaries_dir+"model_valid_score_"+self.model_file_name+"_epoch_"+str(iteration), "wb") as f:
                    pickle.dump(valid_score, f)
                print("Model validation score saved")
            except:
                pass

            if valid_score <= best_valid_score:
                best_valid_score = valid_score
                #best_model = 
                best_epoch = iteration
            elif (iteration-best_epoch <= self.patience):
                pass
            else:
                print("Stopped early at epoch: " + str(iteration))
                break


        best_model = keras.models.load_model(self.model_save_location+self.model_file_name+"_epoch_"+str(best_epoch))
        best_model.save(model_save_location+model_file_name+"_bestValidScore")

        print("Best epoch: " + str(best_epoch) + "   validation score: " + str(best_valid_score))

        print("Testing")
        testDataGen = self.endo_reader.generator_for_autotrain(self.batch_size_m, self.num_steps, "test")
        test_score = best_model.evaluate_generator(testDataGen, base_size_limit*self.trainValTestSplit[2])
        print("Test score: " + str(test_score))


        test_set = self.endo_reader.testSet

        with open(self.summaries_dir+self.model_file_name+"_testSet.p", "wb") as f:
            pickle.dump(test_set, f)
            print("Model test set saved")

        print("Done!!!")

def main(argv):
    my_lstm = keras_endoLSTM(argv)
    my_lstm.run_model(my_lstm.model)

if __name__ == "__main__":
    main(sys.argv)
