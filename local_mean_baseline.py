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
predictions_fn = "heart_rate_local_mean_baseline"
targetVars = ["derived_speed"]

modelRunIdentifier = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")

#Compute on new metadata with excised data removed
endoReader_2=dataInterpreter(fn="../multimodalDBM/endomondoHR_proper.json", allowMissingData=True, scaleVals=False, trimmed_workout_length=trimmed_workout_length)
endoReader_2.buildDataSchema(["sport", "heart_rate", "derived_speed"], targetVars, trainValTestSplit=trainValTestSplit)
test_gen = endoReader_2.endoIteratorSupervised(batch_size, num_steps, 'test')


def compute_workout_mean_error(data_gen):
    test_scores = []
    for i, (inputs, targets) in enumerate(data_gen):
        if i%1000 == 0:
            print("Workout mean baseline computed on " + str(i) + " workouts so far")
            workout_mean = np.mean(targets[0])
        test_scores.append(np.mean(np.square(np.subtract(workout_mean, targets[0]))))
    return test_scores

test_scores = compute_workout_mean_error(test_gen)

#Save test scores
with open(predictions_fn+modelRunIdentifier+".p", "wb") as f:
    pickle.dump(test_scores, f)
    
print("MSE using workout mean " + targetVar + " as a baseline " + str(np.mean(test_scores)))
        
        