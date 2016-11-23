from __future__ import division
import numpy as np
import ijson
import pickle
import os
import jsonReader
from haversine import haversine

#Need to:
        #0. Scan the file when producing metadata and create a list of data point indices
        #1. Load the list of data point indices
        #2. Randomly permute the list of data point indices
        #3. Split the list of data point indices into training, validation, and test sets
        #4. Provide a method to get the next data point in the list (and either globally save the position of the list or save it implicitly in a generator)
        #5. Provide a method to reset this list position (for a new epoch)
        #6. Rewrite the endoIterator method to respect randomization (copy the random access iterator code from the ptb reader example)

class dataInterpreter(object):
    def __init__(self, fn="endomondoHR_proper.json", attributes=None, allowMissingData=True, scaleVals=True):
        self.dataFileName=fn#Will eventually replace this with a data folder name
        self.dataFile=open(self.dataFileName, 'r')
        self.MetaDataLoaded=False
        self.dataSchemaLoaded=False
        #self.currentDataPoint=None
        self.dataPointPosition=0
        self.attIgnore=['id','url']#Attributes to ignore when building metadata
        self.metaDataFn=fn[0:len(fn)-5]+"_metaData.p"
        self.allowMissingData=allowMissingData
        self.scaleVals=scaleVals
        #self.valTestSplit=(.1,.1)
        if attributes is not None:
            self.buildDataSchema(attributes)
    
    def buildDataSchema(self, attributes, trainValTestSplit=(.8,.1,.1)):
        self.buildMetaData()
        self.splitForValidation(trainValTestSplit)
        #self.newEpoch()#Reset all indices and counters
        self.attributes=attributes
        dataDimSum=0
        for att in self.attributes:
            dataDimSum=dataDimSum+self.encodingLengths[att]
        self.dataDim=dataDimSum
        self.dataSchemaLoaded=True
    
    def createSequentialGenerator(self): #Define a new data generator
        filename = self.dataFileName
        self.f=open(filename, 'r')
        objects = ijson.items(self.f, 'users.item')
        self.dataObjects=objects
        return self.dataObjects
    
    def dataGenerator(self, dataSetOrder):
        for dp_index in dataSetOrder:
            fileIndices = self.dataPointIndices[dp_index]
            potentialNextDataPoint=jsonReader.getDataPoint(fileIndices, self.dataFile)
            
            """
            if self.allowMissingData==False:
                #Check if the next data point contains all the requested attributes
                for i, att in enumerate(self.attributes):
                    #print(att)
                    try:
                        test=self.currentDataPoint[att]
                    except:
                        print("Skipping data point because it lacks attribute: " + att)
                        #print("Skipping data point because it lacks attribute")
                        return self.getNextDataPoint() #Try the next one instead
            """
            yield potentialNextDataPoint #returns next data point
    
    def randomizeDataOrder(self, dataIndices):
        return np.random.permutation(dataIndices)

    #def getNextDataPoint(self):
    #        jsonReader.getDataPoint(index, dataFile)
    #    return dataPoint

    def getNextDataPointSequential(self):
        try: #If there is a generator already defined
            objects=self.dataObjects
        except: #Otherwise create a new one
            #Creating new generator
            objects=self.createSequentialGenerator()
        nextDataPoint=self.__convert(objects.next())

        return nextDataPoint
    
    #def newEpoch(self):
        # A convenience function for reseting the data loader to start a new epoch
        #self.currentDataPoint = None
    #    self.dataPointPosition = 0  # The position within a data point (within an exercise)       
    
    
    def batchIteratorSupervised(self, batch_size, trainValidTest, targetAtt):
        #Performs the same job as the batch iterator, but with one of the attributes separated as the supervision signal
        inputAttributes =[x for x in self.attributes if x != targetAtt]
        inputDataDim=self.getInputDim(targetAtt)
        targetDataDim=self.getTargetDim(targetAtt)

        if trainValidTest == 'train':
            self.trainingOrder = self.randomizeDataOrder(self.trainingSet)
            dataGen = self.dataGenerator(self.trainingOrder)
        elif trainValidTest == 'valid':
            self.validationOrder = self.randomizeDataOrder(self.validationSet)
            dataGen = self.dataGenerator(self.validationOrder)
        elif trainValidTest == 'test':
            self.testOrder = self.randomizeDataOrder(self.testSet)
            dataGen = self.dataGenerator(self.testOrder)
        else:
            raise (Exception("Invalid dataset type. Must be 'train', 'valid', or 'test'"))

        if self.dataSchemaLoaded == False:
            raise (RuntimeError("Need to load a data schema"))

        inputDataBatch = np.zeros((batch_size, inputDataDim))
        # inputDataDim is the total concatenated length of the data at each time point (for all input attributes)
        targetDataBatch = np.zeros((batch_size, targetDataDim))


        # if currentDataPoint is None: #If starting an epoch, grab the first data point
        currentDataPoint = dataGen.next()
        dataPointPosition = 0
        currentDataPointLength = self.getDataPointLength(currentDataPoint)
        moreData = True
        currentDerivedData={}
        while moreData:
            for i in range(batch_size):
                # Need code for getting the current data point and iterating through it until the end of it...
                # if end of data point:
                # currentPoint = next data point
                dataList = []  # A mutable data structure to allow us to construct the data instance...
                if dataPointPosition == currentDataPointLength:  # Check to see if new data point is needed
                    try:
                        currentDataPoint = dataGen.next()
                    except:  # If there is no more data, return what you have
                        moreData = False
                        yield [inputDataBatch, targetDataBatch]  # May need to pad this??
                    currentDataPointLength = self.getDataPointLength(currentDataPoint)
                    currentDerivedData = {} #Reset the derived data dictionary
                    dataPointPosition = 0
                for j, att in enumerate(inputAttributes):
                    if self.isDerived[att]:
                        #handle the derived variables
                        if att in currentDerivedData.keys():
                            #Use the data from the current data point position
                            attDataPoint=currentDerivedData[att]
                            attData = attDataPoint[dataPointPosition]
                        else:
                            #Generate the data and then use the data from the current data point position which should be 0
                            currentDerivedData[att] = self.deriveData(att, currentDataPoint)
                            attDataPoint=currentDerivedData[att]
                            attData = attDataPoint[dataPointPosition]
                    else:
                        if self.isSequence[att]:  # Need to limit the sequence to the end of the batch...
                            # Put the sequence attributes in their proper positions in the tensor array
                            # These are numeric encoding schemes.
                            attData = currentDataPoint[att][
                                dataPointPosition]  # Get the next entry in the attribute sequence for the current data point
                        else:
                            # Put the context attributes in their proper positions in the tensor array
                            # These are a one-hot encoding schemes except in the case of "age" and the like
                            if self.isNominal[att]:  # Checks whether the data is nominal
                                attData = self.oneHot(currentDataPoint, att)  # returns a list
                            else:
                                attData = currentDataPoint  # Handles ordinal and numeric data

                    scaledAttData = self.scaleData(attData, att)  # Rescales data if needed
                    if self.isList(scaledAttData):
                        dataList.extend(scaledAttData)
                    else:
                        dataList.append(scaledAttData)
                #Now do the same for the target attribute
                if self.isSequence[targetAtt]:  # Need to limit the sequence to the end of the batch...
                    # Put the target attribute in its proper positions in the tensor array
                    # These are numeric encoding schemes.
                    attData = currentDataPoint[targetAtt][
                        dataPointPosition]  # Get the next entry in the attribute sequence for the current data point
                else:
                    # Put the context attributes in their proper positions in the tensor array
                    # These are a one-hot encoding schemes except in the case of "age" and the like
                    if self.isNominal[targetAtt]:  # Checks whether the data is nominal
                        attData = self.oneHot(currentDataPoint, targetAtt)  # returns a list
                    else:
                        attData = currentDataPoint  # Handles ordinal and numeric data

                scaledTargetAttData = self.scaleData(attData, targetAtt)  # Rescales data if needed
                targetDataBatch[i,:] = scaledTargetAttData#Add the target data for the current data point to the full list

                #Check length of input data vector
                if len(dataList) == inputDataDim:
                    inputDataBatch[i, :] = dataList
                else:
                    print("Data list length: " + str(len(dataList)))
                    print("Data schema length: " + str(len(inputDataDim)))
                    raise (ValueError("Data is not formatted according to the schema"))

                dataPointPosition = dataPointPosition + 1

            yield [inputDataBatch, targetDataBatch]
            
    def dataIteratorSupervised(self, trainValidTest, targetAtt):
        #Performs the same job as the batch iterator, but with one of the attributes separated as the supervision signal
        inputAttributes =[x for x in self.attributes if x != targetAtt]
        inputDataDim=self.getInputDim(targetAtt)
        targetDataDim=self.getTargetDim(targetAtt)

        if trainValidTest == 'train':
            self.trainingOrder = self.randomizeDataOrder(self.trainingSet)
            dataGen = self.dataGenerator(self.trainingOrder)
        elif trainValidTest == 'valid':
            self.validationOrder = self.randomizeDataOrder(self.validationSet)
            dataGen = self.dataGenerator(self.validationOrder)
        elif trainValidTest == 'test':
            self.testOrder = self.randomizeDataOrder(self.testSet)
            dataGen = self.dataGenerator(self.testOrder)
        else:
            raise (Exception("Invalid dataset type. Must be 'train', 'valid', or 'test'"))

        if self.dataSchemaLoaded == False:
            raise (RuntimeError("Need to load a data schema"))

        #inputData = np.zeros((batch_size, inputDataDim))
        # inputDataDim is the total concatenated length of the data at each time point (for all input attributes)
        #targetData = np.zeros((batch_size, targetDataDim))


        # if currentDataPoint is None: #If starting an epoch, grab the first data point
        currentDataPoint = dataGen.next()
        dataPointPosition = 0
        currentDataPointLength = self.getDataPointLength(currentDataPoint)
        moreData = True
        currentDerivedData={}
        while moreData:
            # Need code for getting the current data point and iterating through it until the end of it...
            # if end of data point:
            # currentPoint = next data point
            dataList = []  # A mutable data structure to allow us to construct the data instance...
            if dataPointPosition == currentDataPointLength:  # Check to see if new data point is needed
                try:
                    currentDataPoint = dataGen.next()
                except:  # If there is no more data, return what you have
                    moreData = False
                    yield [inputData, targetData]  # May need to pad this??
                currentDataPointLength = self.getDataPointLength(currentDataPoint)
                currentDerivedData = {} #Reset the derived data dictionary
                dataPointPosition = 0
            for j, att in enumerate(inputAttributes):
                if self.isDerived[att]:
                    #handle the derived variables
                    if att in currentDerivedData.keys():
                        #Use the data from the current data point position
                        attDataPoint=currentDerivedData[att]
                        attData = attDataPoint[dataPointPosition]
                    else:
                        #Generate the data and then use the data from the current data point position which should be 0
                        currentDerivedData[att] = self.deriveData(att, currentDataPoint)
                        attDataPoint=currentDerivedData[att]
                        attData = attDataPoint[dataPointPosition]
                else:
                    if self.isSequence[att]:  # Need to limit the sequence to the end of the batch...
                        # Put the sequence attributes in their proper positions in the tensor array
                        # These are numeric encoding schemes.
                        attData = currentDataPoint[att][
                            dataPointPosition]  # Get the next entry in the attribute sequence for the current data point
                    else:
                        # Put the context attributes in their proper positions in the tensor array
                        # These are a one-hot encoding schemes except in the case of "age" and the like
                        if self.isNominal[att]:  # Checks whether the data is nominal
                            attData = self.oneHot(currentDataPoint, att)  # returns a list
                        else:
                            attData = currentDataPoint  # Handles ordinal and numeric data

                scaledAttData = self.scaleData(attData, att)  # Rescales data if needed
                if self.isList(scaledAttData):
                    dataList.extend(scaledAttData)
                else:
                    dataList.append(scaledAttData)
            #Now do the same for the target attribute
            if self.isSequence[targetAtt]:  # Need to limit the sequence to the end of the batch...
                # Put the target attribute in its proper positions in the tensor array
                # These are numeric encoding schemes.
                attTarData = currentDataPoint[targetAtt][
                    dataPointPosition]  # Get the next entry in the attribute sequence for the current data point
            else:
                # Put the context attributes in their proper positions in the tensor array
                # These are a one-hot encoding schemes except in the case of "age" and the like
                if self.isNominal[targetAtt]:  # Checks whether the data is nominal
                    attTarData = self.oneHot(currentDataPoint, targetAtt)  # returns a list
                else:
                    attTarData = currentDataPoint  # Handles ordinal and numeric data

            scaledTargetAttData = self.scaleData(attTarData, targetAtt)  # Rescales data if needed
            targetData = [] #So that we can return it in the same list format as the inputs
            targetData.append(scaledTargetAttData)#Add the target data for the current data point to the full list

            #Check length of input data vector
            if len(dataList) == inputDataDim:
                inputData = dataList
            else:
                print("Data list length: " + str(len(dataList)))
                print("Data schema length: " + str(len(inputDataDim)))
                raise (ValueError("Data is not formatted according to the schema"))

            dataPointPosition = dataPointPosition + 1

            yield [inputData, targetData]


    def batchIterator(self, batch_size, trainValidTest):
        #Returns a tensorflow tensor (a numpy array) containing a batch of data
        #Can be used directly for feed or to preprocess for additional efficiency
        
        #Currently does not explicitly separate exercise routines. 
        #Can be augmented with a variable that captures end and begnning of a routine if this helps.
        
        if trainValidTest=='train':
            self.trainingOrder = self.randomizeDataOrder(self.trainingSet)
            dataGen=self.dataGenerator(self.trainingOrder)
        elif trainValidTest=='valid':
            self.validationOrder = self.randomizeDataOrder(self.validationSet)
            dataGen=self.dataGenerator(self.validationOrder)
        elif trainValidTest=='test':
            self.testOrder = self.randomizeDataOrder(self.testSet)
            dataGen=self.dataGenerator(self.testOrder)
        else:
            raise(Exception("Invalid dataset type. Must be 'train', 'valid', or 'test'"))
        
        if self.dataSchemaLoaded==False:
            raise(RuntimeError("Need to load a data schema"))
        
        dataBatch = np.zeros((batch_size, self.dataDim))
        #self.dataDim is the total concatenated length of the data at each time point (for all attributes)
                
        #if currentDataPoint is None: #If starting an epoch, grab the first data point
        currentDataPoint=dataGen.next()
        dataPointPosition=0
        currentDataPointLength=self.getDataPointLength(currentDataPoint)
        moreData=True
        while moreData:
            for i in range(batch_size):
                #Need code for getting the current data point and iterating through it until the end of it...
                #if end of data point:
                    #currentPoint = next data point
                dataList = [] #A mutable data structure to allow us to construct the data instance...
                if dataPointPosition==currentDataPointLength: #Check to see if new data point is needed
                    try:
                        currentDataPoint=dataGen.next()
                    except: #If there is no more data, return what you have
                        moreData=False
                        yield dataBatch #May need to pad this??
                    currentDataPointLength=self.getDataPointLength(currentDataPoint)
                    dataPointPosition=0
                for j, att in enumerate(self.attributes):
                    if self.isSequence[att]: #Need to limit the sequence to the end of the batch...
                        #Put the sequence attributes in their proper positions in the tensor array
                        #These are numeric encoding schemes.
                        attData=currentDataPoint[att][dataPointPosition]#Get the next entry in the attribute sequence for the current data point
                    else:
                        #Put the context attributes in their proper positions in the tensor array
                        #These are a one-hot encoding schemes except in the case of "age" and the like
                        if self.isNominal[att]:#Checks whether the data is nominal
                            attData = self.oneHot(currentDataPoint, att) #returns a list
                        else:
                            attData = currentDataPoint #Handles ordinal and numeric data

                    scaledAttData=self.scaleData(attData, att)#Rescales data if needed
                    if self.isList(scaledAttData):
                        dataList.extend(scaledAttData)
                    else:
                        dataList.append(scaledAttData)           
                if len(dataList)==self.dataDim:
                    dataBatch[i,:]=dataList
                else:
                    print("Data list length: " + dataList)
                    print("Data schema length: " + self.dataDim)
                    raise(ValueError("Data is not formatted according to the schema"))

                dataPointPosition=dataPointPosition+1

            yield dataBatch

    def endoIteratorSupervised(self, batch_size, num_steps, trainValidTest, targetAtt):
            #Does the same thing as the endoIterator, except for a model with separate supervised targets (targets are not the next element in the sequence)

            batchGen = self.dataIteratorSupervised(trainValidTest, targetAtt)

            data_len = self.numDataPoints
            batch_len = data_len // batch_size
            epoch_size = (batch_len - 1) // num_steps

            inputDataDim = self.getInputDim(targetAtt)
            targetDataDim = self.getTargetDim(targetAtt)

            if epoch_size == 0:
                raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
                

            nextRow=[next(batchGen) for x in range(num_steps)]#Fill the first row (with both inputs and targets)
            #print(nextRowTargets)
            for i in range(epoch_size):
                inputData = np.zeros([batch_size*num_steps, num_steps, inputDataDim])
                targetData = np.zeros([batch_size*num_steps, num_steps, targetDataDim])
                for j in range(batch_size):
                    for k in range(num_steps):
                        #print(" j: " + str(j) + " k: " + str(k))
                        currentRow=nextRow

                        inputData[((j*num_steps)+k), :, :]  = [currentRow[x][0] for x in range(num_steps)]
                        targetData[((j*num_steps)+k), :, :] = [currentRow[x][1] for x in range(num_steps)]

                        nextRow[0:(num_steps-1)] = currentRow[1:num_steps]
                        nextRow[num_steps-1]=batchGen.next()

                x = inputData
                y = targetData
                yield (x, y)

    
    def endoIterator(self, batch_size, num_steps, trainValidTest):
        #Returns a batch-wise generator for the endomondo data with inputs as the current element in the sequence and targets as the next element in the sequence

        batchGen = self.batchIterator(batch_size*(num_steps+1), trainValidTest)

        data_len = self.numDataPoints
        batch_len = data_len // batch_size
        epoch_size = (batch_len - 1) // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        #For these guys, the labels are simply the next sequence. This is to train the model to reprodue the text.
        #Since I am not really trying to do this, I should generate the labels seperately.
        #However, I might find that training the net this way (to predict the sequence) and then transplanting the weights into the full model might be useful...
        """for i in range(epoch_size):
            batchData=self.nextBatch(batch_size)
            data = np.zeros([batch_size, batch_len, self.dataDim])
            for j in range(batch_size):
                data[j] = batchData[batch_len * j:batch_len * (j + 1)]
            x = data[:, i*num_steps:(i+1)*num_steps]
            y = data[:, i*num_steps+1:(i+1)*num_steps+1]
            yield (x, y)"""
        
        #The code below is not ideal because it trains everything in order whereby each batch is comprised sequentially of the data
        #It should be OK for basic testing, however.
        #and it may miss some transitions (not sure)
        #print( epoch_size)
        for i in range(epoch_size):
            batchData=batchGen.next()
            #print(batchData.shape)
            data = np.zeros([batch_size, num_steps+1, self.dataDim])
            for j in range(batch_size):
                data[j,:,:] = batchData[(num_steps * j):((num_steps * (j + 1))+1),:]
            #print(data.shape)
            #x = data[:, i*num_steps:((i+1)*num_steps),:]
            x = data[:, 0:num_steps, :]
            #y = data[:, i*(num_steps+1):((i+1)*num_steps+1),:]
            y = data[:, 1:(num_steps+1), :]
            yield (x, y)
        
    
    def splitForValidation(self, valTestSplit):
        #Construct seperate data files for the training, test, and validation data
        self.numDataPoints
        trainingSetSize=int(round(self.numDataPoints*valTestSplit[0]))
        validationSetSize=int(round(self.numDataPoints*valTestSplit[1]))
        testSetSize=int(round(self.numDataPoints*valTestSplit[2]))
        randomOrder=self.randomizeDataOrder(self.numDataPoints)
        
        self.trainingSet=randomOrder[0:trainingSetSize]
        self.validationSet=randomOrder[trainingSetSize:trainingSetSize+validationSetSize]
        self.testSet=randomOrder[trainingSetSize+validationSetSize:trainingSetSize+validationSetSize+testSetSize]
        
        #print("training set size:" + str(len(self.trainingSet))
        #print("validation set size:" + str(len(self.validationSet))
        #print("test set size:" + str(len(self.testSet)))

    def deriveData(self, att, currentDataPoint):
        if att=='time_elapsed':
            #Derive the time elapsed data sequence
            timestamps=currentDataPoint['timestamp']
            initialTime=timestamps[0]
            return [x-initialTime for x in timestamps]
            #Going to need to scale these appropriately in the scaling function
        elif att=='distance':
            #Derive the distance data sequence
            lats=currentDataPoint['latitude']
            longs=currentDataPoint['longitude']
            indices=range(1, len(lats)) #Creates a list of indices from 1 to the length of the seuqnces to index all elements but the first (element zero)
            distances=[0]
            distances.extend([haversine([lats[i-1],longs[i-1]], [lats[i],longs[i]]) for i in indices]) #Gets distance traveled so far in kilometers
            return distances
        elif att=='new_workout':
            workoutLength=len(currentDataPoint['timestamp'])#This is likely always equal to 500
            newWorkout=np.zeros(workoutLength)
            newWorkout[0]=1 #Add the signal
            return newWorkout
        else:
            raise(Exception("No such derived data attribute"))
            
    def scaleData(self, data, att):
        #This function provides optional rescaling of the data for optimal neural network performance. 
        #It can either be run online or offline w/ results stored in a preprocessed data file (more effecient)
        if self.scaleVals:
            if att=="speed":
                scaledData=data
                return scaledData
            elif att=="heart_rate":
                scaledData=data/250.0 #This will be replaced with an auto-ranging version
                return scaledData
            elif att=="altitude":
                scaledData=float(data)/10000.0 #This will be replaced with an auto-ranging version
                return scaledData
            else:
                return data
        else:
            return data
        
    def __convert(self, unicData): #Converts the unicode text in a dictionary to ascii
        #Shamelessly lifted from http://stackoverflow.com/questions/13101653/python-convert-complex-dictionary-of-strings-from-unicode-to-ascii
        if isinstance(unicData, dict):
            return {self.__convert(key): self.__convert(value) for key, value in unicData.iteritems()}
        elif isinstance(unicData, list):
            return [self.__convert(element) for element in unicData]
        elif isinstance(unicData, unicode):
            return unicData.encode('utf-8')
        else:
            return unicData
        
    def getDataPointLength(self, dataPoint):
        #Checks a single attribute. If the length of all sequence attributes is not equal, additional code will need to be written...
        return len(dataPoint["heart_rate"])#tries "heart_rate"
    
    def isList(self, attData):
        #checks whether the variable attData is a list and returns true or false
        return isinstance(attData, list)
        #might want to try isSubclass(attData, list) if this doesn't work...
    
    def buildEncoder(self, classLabels):
        #Constructs a dictionary that maps each class label to a list (encoding scheme) where one entry in the list is 1 and the remainder are 0
        encodingLength=classLabels.size
        encoder={}
        for i, label in enumerate(classLabels):
            encoding=[0] * encodingLength
            encoding[i]=1
            encoder[label]=encoding
        return encoder
    
    def getDataLabels(self, data, dataClass):
        #The "data" argument is in the same format as is returned by "getNdatapoints"
        #If there is a use case that involves finding all the possible labels for a given class, a seperate function should be written to save memory usage...
        class_labels = [col[dataClass] for col in data]
        return np.unique(np.array(class_labels))
    
    def writeSummaryFile(self):
        metaDataForWriting=metaDataEndomondo(self.numDataPoints, self.encodingLengths, self.oneHotEncoders, 
                                             self.isSequence, self.isNominal, self.isDerived, self.dataPointIndices)
        with open(self.metaDataFn, "wb") as f:
            pickle.dump(metaDataForWriting, f)

        #pickle.dump(metaDataForWriting, open(self.metaDataFn, "wb"))
        print("Summary file written")
        
    def loadSummaryFile(self):
        try:
            print("Loading metadata")
            with open(self.metaDataFn, "rb") as f:
                metaData = pickle.load(f)
                
            #metaData=pickle.load(open(self.metaDataFn, "rb"))
        except:
            raise(IOError("Metadata file: " + self.metaDataFn + " not in valid pickle format"))
        self.numDataPoints=metaData.numDataPoints
        self.encodingLengths=metaData.encodingLengths
        self.oneHotEncoders=metaData.oneHotEncoders
        #self.dataDim=metaData.dataDim
        self.isSequence=metaData.isSequence
        self.isNominal=metaData.isNominal
        self.isDerived=metaData.isDerived
        self.dataPointIndices=metaData.dataPointIndices
        print("Metadata loaded")
        
    def buildMetaData(self):
        #Takes a list of attributes and the current datafile and constructs a schema for the data to be input into the RNN.
        if os.path.isfile(self.metaDataFn):#If a summary file exists
            self.loadSummaryFile()#Load that summary file and use it to capture all the necessary info
        else:
            print("Building data schema")
            #Build such a summary file by running through the full dataset and capturing the necessary statistics
            self.isSequence={'altitude':True, 'gender':False, 'heart_rate':True, 'id':False, 'latitude':True, 'longitude':True,
                             'speed':True, 'sport':False, 'timestamp':True, 'url':False, 'userId':False, 'time_elapsed': True, 
                             'distance':True, 'new_workout':True}#Handcoded
            self.isNominal={'altitude':False, 'gender':True, 'heart_rate':False, 'id':True, 'latitude':False, 'longitude':False,
                            'speed':False, 'sport':True, 'timestamp':False, 'url':True, 'userId':True, 'time_elapsed': False,
                            'distance':False, 'new_workout':False}#Handcoded
            self.isDerived={'altitude':False, 'gender':False, 'heart_rate':False, 'id':False, 'latitude':False, 'longitude':False,
                   'speed':False, 'sport':False, 'timestamp':False, 'url':False, 'userId':False, 'time_elapsed': True,
                            'distance':True, 'new_workout':True}#Handcoded
            allDataClasses=['altitude', 'gender', 'heart_rate', 'id', 'latitude', 'longitude',
                   'speed', 'sport', 'timestamp', 'url', 'userId', 'time_elapsed', 'distance', 'new_workout']
            dataClasses=[x for x in allDataClasses if x not in self.attIgnore]#get rid of the attributes that we are ignoring
            #self.newEpoch()#makes sure to reset things
            moreData=True
            classLabels={}
            numDataPoints=0
            while moreData:
                if numDataPoints%1000==0:
                    print("Currently at data point " + str(numDataPoints))
                try:
                    currData=[self.getNextDataPointSequential()]
                    #dataClasses = self.getDataClasses(currData)#This could be removed to make it more effecient
                    for datclass in dataClasses:
                        if self.isDerived[datclass] != True:
                            if self.isNominal[datclass]: #If it is nominal data
                                if self.isSequence[datclass]:
                                    raise(NotImplementedError("Nominal data types for sequences have not yet been implemented"))
                                dataClassLabels=self.getDataLabels(currData, datclass)
                                if classLabels.get(datclass) is None: #If it is the first step
                                    classLabels[datclass]=dataClassLabels
                                else:
                                    #print(np.concatenate(dataClassLabels,classLabels[datclass]))
                                    classLabels[datclass]=np.unique(np.concatenate([dataClassLabels,classLabels[datclass]]))
                            else:
                                if self.isSequence[datclass]!=True:
                                    #If is it nominal and not a sequence
                                    raise(NotImplementedError("Non-nominal data types for non-sequences have not yet been implemented"))
                    numDataPoints=numDataPoints+1
                except:
                    moreData=False
                    print("Stopped at " + str(numDataPoints) + " data points")
                #if numDataPoints>10000:#For testing
                #    moreData=False#For testing
            
            oneHotEncoders={}
            encodingLengths={}
            dataDim=0
            for datclass in dataClasses:
                if self.isSequence[datclass]==False:
                    oneHotEncoders[datclass]=self.buildEncoder(classLabels[datclass])
                    encodingLengths[datclass]=classLabels[datclass].size
                    #dataDim=dataDim+encodingLengths[datclass]
                else:
                    if self.isNominal[datclass]:
                        raise(NotImplementedError("Nominal data types for sequences have not yet been implemented"))
                    else:
                        encodingLengths[datclass]=1
                        #dataDim=dataDim+1
            print("Getting data indices")
            dataPointIndices=jsonReader.getDataIndices(self.dataFileName)
            
            #Set all of the summary information to self properties
            self.numDataPoints=numDataPoints
            self.encodingLengths=encodingLengths#A dictionary that maps attributes to the lengths of their vector encoding schemes
            self.oneHotEncoders=oneHotEncoders#A dictionary of dictionaries where the outer dictionary maps attributes to encoding schemes and where each encoding scheme is a dictionary that maps attribute values to one hot encodings
            #self.dataDim=dataDim#The sum of all the encoding lengths for the relevant attributes
            #self.isSequence=#A dictionary that returns whether an attribute takes the form of a sequence of data
            #self.isNominal=#A dictionary that returns whether an attribute is nominal in form (neither numeric nor ordinal)
            self.dataPointIndices=dataPointIndices
            
            #Save that summary file so that it can be used next time
            self.writeSummaryFile()
        self.MetaDataLoaded=True 
        
    def oneHot(self, dataPoint, att):
        #Takes the current data point and the attribute type and uses the data schema to provide the one-hot encoding for the variable
        dataValue=dataPoint[att]       
        #Use a stored schema dictionary to return the correct encoding scheme for the attribute (an encoding scheme is also a dictionary)
        encoder=self.oneHotEncoders[att]
        #Use this encoding scheme to get the encoding
        encoding=encoder[dataValue]
        return encoding

    def getInputDim(self, targetAtt):
        return self.dataDim - self.encodingLengths[targetAtt]

    def getTargetDim(self, targetAtt):
        return self.encodingLengths[targetAtt]
    
    
class metaDataEndomondo(object):
    #For disk storage of metadata
    #Meant to be pickled and unpickled
    def __init__(self, numDataPoints, encodingLengths, oneHotEncoders, isSequence, isNominal, isDerived, dataPointIndices):
        self.numDataPoints=numDataPoints
        self.encodingLengths=encodingLengths
        self.oneHotEncoders=oneHotEncoders
        #self.dataDim=dataDim
        self.isSequence=isSequence
        self.isNominal=isNominal
        self.isDerived=isDerived
        self.dataPointIndices=dataPointIndices