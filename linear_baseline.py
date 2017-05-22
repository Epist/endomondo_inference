from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sklearn
from sklearn import linear_model
import numpy as np
from data_interpreter_Keras_multiTarget import dataInterpreter, metaDataEndomondo
import warnings
import pickle
from sklearn.externals import joblib
import time
import datetime


model_fn = "baselines/linear_baseline_speed_noTime_noDist"
predictions_fn = "baselines/linear_baseline_speed_noTime_noDist"

endoFeatures = ["heart_rate", "new_workout", "gender", "sport", "userId", "altitude", "derived_speed"] #["heart_rate", "new_workout", "gender", "sport", "userId", "altitude", "distance", "derived_speed", "time_elapsed"]
targetAtts = ['derived_speed']
modelRunIdentifier = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")

trainValTestSplit=(0.8, 0.1, 0.1)
trimmed_workout_length=450 #450
zMultiple = 5
scaleVals=False
scaleTargets=False

trainValTestFN = "logs/k'eras/keras__noSport_hrTarget" #The filename root from which to load the train valid test split
endoReader=dataInterpreter(fn="../multimodalDBM/endomondoHR_proper.json", allowMissingData=True, scaleVals=scaleVals, trimmed_workout_length=trimmed_workout_length, scaleTargets=False)
endoReader.buildDataSchema(endoFeatures, targetAtts, trainValTestSplit=trainValTestSplit, zMultiple = zMultiple, trainValidTestFN = trainValTestFN)
batch_size=1
num_steps=1
train_iter = endoReader.endoIteratorSupervised(batch_size, num_steps, "train")



models = []
models.append(linear_model.LinearRegression())




def fit_on_dataset(models, train_gen):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i, (inputs, targets) in enumerate(train_gen):
        #for i in range(100):
            #(inputs, targets) = endoIter.next()
            if i%(1000*450) == 0:
                print("Trained on " + str(int(i/450)) + " workouts so far")
            for model in models:
                model.fit(inputs[0], targets[0])
        
print("Training " + str(len(models)) + " linear models")
fit_on_dataset(models, train_iter)
print("Done training")


#Save models
for i, model in enumerate(models):
    joblib.dump(model, model_fn + modelRunIdentifier + "_model" + str(i) + ".p")
print("Models saved")

test_iter = endoReader.endoIteratorSupervised(batch_size, num_steps, "test")

def seqMSE(predictions, targets):
    return np.mean(np.square(np.subtract(predictions, targets)))

def test_lin(models, test_gen):
    workout_test_scores_by_model = []
    for x in range(len(models)):
        workout_test_scores_by_model.append([])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  
        for i, (inputs, targets) in enumerate(test_gen):
            if i%(1000*450) == 0:
                print("Tested on " + str(i/450) + " workouts so far")
                print("Mean MSEs so far: " + str([np.mean(x) for x in workout_test_scores_by_model]))
            for j, model in enumerate(models):
                preds = model.predict(inputs[0])
                workout_test_scores_by_model[j].append(seqMSE(preds, targets[0]))
    return workout_test_scores_by_model

print("Testing Linear Model") 
all_models_test_scores = test_lin(models, test_iter)
print("Done testing")

#Save models
#with open(predictions_fn+modelRunIdentifier+"MODELS.p", "wb") as f:
#    pickle.dump(models, f)

#Save test scores
with open(predictions_fn+modelRunIdentifier+".p", "wb") as f:
    pickle.dump(all_models_test_scores, f)
    
for i, model in enumerate(models):
    print("Mean linear model MSE on test set for model " + str(i) + " : " + str(np.mean(all_models_test_scores[i])))

