from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from data_interpreter_Keras_multiTarget import dataInterpreter, metaDataEndomondo
import time
import datetime
import pickle

trimmed_workout_length=450
batch_size=1
num_steps=trimmed_workout_length
trainValTestSplit=(0, 0, 1)
predictions_fn = "speed_global_mean_baseline"
targetVar = "derived_speed"

modelRunIdentifier = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")

#Compute on new metadata with excised data removed
endoReader_2=dataInterpreter(fn="../multimodalDBM/endomondoHR_proper.json", allowMissingData=True, scaleVals=False, trimmed_workout_length=trimmed_workout_length)
endoReader_2.buildDataSchema(["sport", "heart_rate", "derived_speed"], [targetVar], trainValTestSplit=trainValTestSplit)
test_gen = endoReader_2.endoIteratorSupervised(batch_size, num_steps, 'test')


heart_rate_mean = endoReader_2.variableMeans[targetVar]
#dspeed_mean = endoReader_2.variableMeans["derived_speed"]


def compute_global_mean_error(global_mean, data_gen):
    test_scores = []
    for i, (inputs, targets) in enumerate(data_gen):
        if i%1000 == 0:
            print("Mean baseline computed on " + str(i) + " workouts so far")
        test_scores.append(np.mean(np.square(np.subtract(heart_rate_mean, targets[0]))))
    return test_scores

test_scores = compute_global_mean_error(heart_rate_mean, test_gen)

#Save test scores
with open(predictions_fn+modelRunIdentifier+".p", "wb") as f:
    pickle.dump(test_scores, f)
    
print("MSE using global mean " + targetVar + " as a baseline " + str(np.mean(test_scores)))
        
        
        