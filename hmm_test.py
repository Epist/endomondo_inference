from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from data_interpreter_Keras_multiTarget import dataInterpreter, metaDataEndomondo
import warnings
from hmmlearn import hmm
import pickle
from sklearn.externals import joblib
import time
import datetime


model_fn = "baselines/hmm_test_multi_model_speed"
predictions_fn = "baselines/hmm_test_multi_preds_speed"

targetAtts=['derived_speed']
modelRunIdentifier = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")

trainValTestSplit=[0.8, 0.1, 0.1]
trimmed_workout_length=450

trainValTestFN = "logs/keras/keras__noSport_hrTarget" #The filename root from which to load the train valid test split

endoReader=dataInterpreter(fn="../multimodalDBM/endomondoHR_proper.json", allowMissingData=True, scaleVals=False, trimmed_workout_length=trimmed_workout_length, scaleTargets=False)
endoReader.buildDataSchema(["heart_rate", "derived_speed"], targetAtts, trainValTestSplit=trainValTestSplit, zMultiple = 0, trainValidTestFN = trainValTestFN)
batch_size=1
num_steps=endoReader.trimmed_workout_length
train_iter = endoReader.endoIteratorSupervised(batch_size, num_steps, "train")

#Train unsupervised HMM

#Use fn="../multimodalDBM/endomondoHR_proper_copy.json"

#Train a lot of models at once to capitalize on the fact that much of the processing involves generating the data
models = []
models.append(hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100))
models.append(hmm.GaussianHMM(n_components=5, covariance_type="full", n_iter=100))
#models.append(hmm.GaussianHMM(n_components=10, covariance_type="full", n_iter=100))
#models.append(hmm.GaussianHMM(n_components=3, covariance_type="spherical", n_iter=100))
#models.append(hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100))
#models.append(hmm.GaussianHMM(n_components=3, covariance_type="tied", n_iter=100))




def fit_hmm_on_dataset(models, train_gen):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i, (inputs, targets) in enumerate(train_gen):
        #for i in range(100):
            #(inputs, targets) = endoIter.next()
            if i%1000 == 0:
                print("Trained on " + str(i) + " workouts so far")
            for model in models:
                model.fit(targets[0])
        
print("Training " + str(len(models)) + " hmm models")
fit_hmm_on_dataset(models, train_iter)
print("Done training")


#Save HMM models
for i, model in enumerate(models):
    joblib.dump(model, model_fn + modelRunIdentifier + "_model" + str(i) + ".p")
print("Models saved")

test_iter = endoReader.endoIteratorSupervised(batch_size, num_steps, "test")

def seqMSE(predictions, targets):
    return np.mean(np.square(np.subtract(predictions, targets)))

def test_hmm(models, test_gen):
    workout_test_scores_by_model = []
    for i in range(len(models)):
        workout_test_scores_by_model.append([])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  
        for i, (inputs, targets) in enumerate(test_gen):
            if i%1000 == 0:
                print("Tested on " + str(i) + " workouts so far")
                print("Mean MSEs so far: " + str([np.mean(x) for x in workout_test_scores_by_model]))
            for i, model in enumerate(models):
                x,z = model.sample(num_steps)
                workout_test_scores_by_model[i].append(seqMSE(x, targets[0]))
    return workout_test_scores_by_model

print("Testing HMM") 
all_models_test_scores = test_hmm(models, test_iter)
print("Done testing")
                     

#Save test scores
with open(predictions_fn+modelRunIdentifier+".p", "wb") as f:
    pickle.dump(all_models_test_scores, f)
    
for i, model in enumerate(models):
    print("Mean hmm MSE on test set for model " + str(i) + " : " + str(np.mean(all_models_test_scores[i])))
            
            

