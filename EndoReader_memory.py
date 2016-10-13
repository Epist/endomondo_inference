from __future__ import division
# Interprets the repaired json for the endomondo data

# Might want to check for the presence of all attributes

# Need to provide separate training, validation, and test data
# Need to precompute max and min of numerical variables for scaling
# Need to impliment scaling

import numpy as np
import ijson
import pickle
import os



class dataInterpreter(object):
    """Going to want to impliment a thing that fixes the data (attribute) schema
    once it's set both for effeciency purposes and for safety purposes.
    This means that this object gets a list of attributes to focus on that is immutable
    and always retrieves data w.r.t. this attribute list.
    This attribute list should also be readable so that programs that use this class
    can configure things based on information about how the variables are handled in this reader.
    """

    """The most user-relevant functions are buildDataSchema, nextBatch, and newEpoch. Use the rest with caution,
    as they may have unanticipated effects (for someone not familiar with this package)"""

    # The dataSet paramater can be set to "train", "valid", or "test" to read from those data sets respectively

    def __init__(self, fn="endomondoHR_proper.json", attributes=None, dataSet="train", allowMissingData=True):
        self.dataFileName = fn  # Will eventually replace this with a data folder name
        self.MetaDataLoaded = False
        self.dataSchemaLoaded = False
        self.currentDataPoint = None
        self.currentDataPointNumber = 0
        self.dataPointPosition = 0
        self.attIgnore = ['id', 'url']  # Attributes to ignore when building metadata
        self.metaDataFn = fn[0:len(fn) - 5] + "_metaData.p"
        self.allowMissingData = allowMissingData
        # self.valTestSplit=(.1,.1)
        if attributes is not None:
            self.buildDataSchema(attributes)

    def createSequentialGenerator(self):  # Define a new data generator
        filename = self.dataFileName
        self.f = open(filename, 'r')
        objects = ijson.items(self.f, 'users.item')
        self.dataObjects = objects
        return self.dataObjects

    def loadAllData(self):
        objects = self.createSequentialGenerator()
        moreData = True
        columns = []
        while moreData is True:
            try:
                columns.append(objects.next())
            except:
                moreData = False
        self.dataSet = self.__convert(columns)  # data as a list of dictionaries where each dictionary represents a particular exercise by a particular user

    """
    def getNDataPoints(self, n):
        try: #If there is a generator already defined
            objects=self.dataObjects
        except: #Otherwise create a new one
            #Creating new generator
            objects=self.createGenerator()
        columns = []
        for i in range(0,n):
            columns.append(objects.next())
        return self.__convert(columns) #returns data as a list of dictionaries where each dictionary represents a particular exercise by a particular user
    """

    def randomizeDataOrder(self, dataIndices):
        return np.random.permutation(dataIndices)

    def dataGenerator(self, dataSet):

        for potentialNextDataPoint in dataSet:
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
            yield potentialNextDataPoint  # returns next data point

    def __convert(self, unicData):  # Converts the unicode text in a dictionary to ascii
        # Shamelessly lifted from http://stackoverflow.com/questions/13101653/python-convert-complex-dictionary-of-strings-from-unicode-to-ascii
        if isinstance(unicData, dict):
            return {self.__convert(key): self.__convert(value) for key, value in unicData.iteritems()}
        elif isinstance(unicData, list):
            return [self.__convert(element) for element in unicData]
        elif isinstance(unicData, unicode):
            return unicData.encode('utf-8')
        else:
            return unicData

    def getDataClasses(self, data):
        class_label_lists = [col.keys() for col in data]
        class_labels = [item for sublist in class_label_lists for item in sublist]
        return np.unique(np.array(class_labels))

    def getDataLabels(self, data, dataClass):
        # The "data" argument is in the same format as is returned by "getNdatapoints"
        # If there is a use case that involves finding all the possible labels for a given class, a seperate function should be written to save memory usage...
        class_labels = [col[dataClass] for col in data]
        return np.unique(np.array(class_labels))

    def getAttributeData(self, data, dataClass):
        # Get list of data points of a given attribute from a data thing (list of dictionaries)
        class_labels = [col[dataClass] for col in data]
        return class_labels

    def buildEncoder(self, classLabels):
        # Constructs a dictionary that maps each class label to a list (encoding scheme) where one entry in the list is 1 and the remainder are 0
        encodingLength = classLabels.size
        encoder = {}
        for i, label in enumerate(classLabels):
            encoding = [0] * encodingLength
            encoding[i] = 1
            encoder[label] = encoding
        return encoder

    def writeSummaryFile(self):
        metaDataForWriting = metaDataEndomondo(self.numDataPoints, self.encodingLengths, self.oneHotEncoders,
                                               self.isSequence, self.isNominal)
        with open(self.metaDataFn, "wb") as f:
            pickle.dump(metaDataForWriting, f)

        # pickle.dump(metaDataForWriting, open(self.metaDataFn, "wb"))
        print("Summary file written")

    def loadSummaryFile(self):
        try:
            print("Loading metadata")
            with open(self.metaDataFn, "rb") as f:
                metaData = pickle.load(f)

                # metaData=pickle.load(open(self.metaDataFn, "rb"))
        except:
            raise (IOError("Metadata file: " + self.metaDataFn + " not in valid pickle format"))
        self.numDataPoints = metaData.numDataPoints
        self.encodingLengths = metaData.encodingLengths
        self.oneHotEncoders = metaData.oneHotEncoders
        # self.dataDim=metaData.dataDim
        self.isSequence = metaData.isSequence
        self.isNominal = metaData.isNominal
        print("Metadata loaded")

    def buildDataSchema(self, attributes):
        self.buildMetaData()
        self.loadAllData()
        self.splitForValidation((.8, .1, .1))
        self.newEpoch()  # Reset all indices and counters and orders
        self.attributes = attributes
        dataDimSum = 0
        for att in self.attributes:
            dataDimSum = dataDimSum + self.encodingLengths[att]
        self.dataDim = dataDimSum
        self.dataSchemaLoaded = True

    def splitForValidation(self, valTestSplit):
        # Construct seperate data files for the training, test, and validation data
        self.numDataPoints
        trainingSetSize = int(round(self.numDataPoints * valTestSplit[0]))
        validationSetSize = int(round(self.numDataPoints * valTestSplit[1]))
        testSetSize = int(round(self.numDataPoints * valTestSplit[2]))
        randomOrder = np.random.permutation(self.numDataPoints)

        self.trainingSet = randomOrder[0:trainingSetSize]
        self.validationSet = randomOrder[trainingSetSize:validationSetSize]
        self.testSet = randomOrder[validationSetSize:testSetSize]


        # Save metadata for each set (probably just need the number of data points...)
        # Write the data files with the appropriate data points (possibly do this progressively within the for loop)

    def buildMetaData(self):
        # Takes a list of attributes and the current datafile and constructs a schema for the data to be input into the RNN.
        if os.path.isfile(self.metaDataFn):  # If a summary file exists
            self.loadSummaryFile()  # Load that summary file and use it to capture all the necessary info
        else:
            print("Building data schema")
            # Build such a summary file by running through the full dataset and capturing the necessary statistics
            self.isSequence = {'altitude': True, 'gender': False, 'heart_rate': True, 'id': False, 'latitude': True,
                               'longitude': True,
                               'speed': True, 'sport': False, 'timestamp': True, 'url': False,
                               'userId': False}  # Handcoded
            self.isNominal = {'altitude': False, 'gender': True, 'heart_rate': False, 'id': True, 'latitude': False,
                              'longitude': False,
                              'speed': False, 'sport': True, 'timestamp': False, 'url': True,
                              'userId': True}  # Handcoded
            allDataClasses = ['altitude', 'gender', 'heart_rate', 'id', 'latitude', 'longitude',
                              'speed', 'sport', 'timestamp', 'url', 'userId']
            dataClasses = [x for x in allDataClasses if
                           x not in self.attIgnore]  # get rid of the attributes that we are ignoring
            self.newEpoch()  # makes sure to reset things
            moreData = True
            classLabels = {}
            numDataPoints = 0
            while moreData:
                if numDataPoints % 1000 == 0:
                    print("Currently at data point " + str(numDataPoints))
                try:
                    currData = self.getNDataPoints(1)
                    # dataClasses = self.getDataClasses(currData)#This could be removed to make it more effecient
                    for datclass in dataClasses:
                        if self.isNominal[datclass]:  # If it is nominal data
                            if self.isSequence[datclass]:
                                raise (
                                NotImplementedError("Nominal data types for sequences have not yet been implemented"))
                            dataClassLabels = self.getDataLabels(currData, datclass)
                            if classLabels.get(datclass) is None:  # If it is the first step
                                classLabels[datclass] = dataClassLabels
                            else:
                                # print(np.concatenate(dataClassLabels,classLabels[datclass]))
                                classLabels[datclass] = np.unique(
                                    np.concatenate([dataClassLabels, classLabels[datclass]]))
                        else:
                            if self.isSequence[datclass] != True:
                                # If is it nominal and not a sequence
                                raise (NotImplementedError(
                                    "Non-nominal data types for non-sequences have not yet been implemented"))
                    numDataPoints = numDataPoints + 1
                except:
                    moreData = False
                    print("Stopped at " + str(numDataPoints) + " data points")
                    # if numDataPoints>10000:#For testing
                    #    moreData=False#For testing

            oneHotEncoders = {}
            encodingLengths = {}
            dataDim = 0
            for datclass in dataClasses:
                if self.isSequence[datclass] == False:
                    oneHotEncoders[datclass] = self.buildEncoder(classLabels[datclass])
                    encodingLengths[datclass] = classLabels[datclass].size
                    # dataDim=dataDim+encodingLengths[datclass]
                else:
                    if self.isNominal[datclass]:
                        raise (NotImplementedError("Nominal data types for sequences have not yet been implemented"))
                    else:
                        encodingLengths[datclass] = 1
                        # dataDim=dataDim+1

            # Set all of the summary information to self properties
            self.numDataPoints = numDataPoints
            self.encodingLengths = encodingLengths  # A dictionary that maps attributes to the lengths of their vector encoding schemes
            self.oneHotEncoders = oneHotEncoders  # A dictionary of dictionaries where the outer dictionary maps attributes to encoding schemes and where each encoding scheme is a dictionary that maps attribute values to one hot encodings
            # self.dataDim=dataDim#The sum of all the encoding lengths for the relevant attributes
            # self.isSequence=#A dictionary that returns whether an attribute takes the form of a sequence of data
            # self.isNominal=#A dictionary that returns whether an attribute is nominal in form (neither numeric nor ordinal)

            # Save that summary file so that it can be used next time
            self.writeSummaryFile()
        self.MetaDataLoaded = True

    def oneHot(self, dataPoint, att):
        # Takes the current data point and the attribute type and uses the data schema to provide the one-hot encoding for the variable
        dataValue = dataPoint[att]
        # Use a stored schema dictionary to return the correct encoding scheme for the attribute (an encoding scheme is also a dictionary)
        encoder = self.oneHotEncoders[att]
        # Use this encoding scheme to get the encoding
        encoding = encoder[dataValue]
        return encoding

    def isList(self, attData):
        # checks whether the variable attData is a list and returns true or false
        return isinstance(attData, list)
        # might want to try isSubclass(attData, list) if this doesn't work...

    def getDataPointLength(self, dataPoint):
        # Checks a single attribute. If the length of all sequence attributes is not equal, additional code will need to be written...
        return len(dataPoint["heart_rate"])  # tries "heart_rate"

    def nextBatch(self, batch_size):
        # Returns a tensorflow tensor (a numpy array) containing a batch of data
        # Can be used directly for feed or to preprocess for additional efficiency

        # Currently does not explicitly separate exercise routines.
        # Can be augmented with a variable that captures end and begnning of a routine if this helps.

        if self.dataSchemaLoaded == False:
            raise (RuntimeError("Need to load a data schema"))

        dataBatch = np.zeros((batch_size, self.dataDim))
        # self.dataDim is the total concatenated length of the data at each time point (for all attributes)

        if self.currentDataPoint is None:  # If starting an epoch, grab the first data point
            self.currentDataPoint = self.getNextDataPoint()
            self.currentDataPointNumber = self.currentDataPointNumber + 1
        currentDataPointLength = self.getDataPointLength(self.currentDataPoint)

        for i in range(batch_size):
            # Need code for getting the current data point and iterating through it until the end of it...
            # if end of data point:
            # currentPoint = next data point
            dataList = []  # A mutable data structure to allow us to construct the data instance...
            if self.dataPointPosition == currentDataPointLength:  # Check to se if new data point is needed
                try:
                    self.currentDataPoint = self.getNextDataPoint()
                    self.currentDataPointNumber = self.currentDataPointNumber + 1
                except:  # If there is no more data, return what you have
                    return dataBatch  # May need to pad this??
                currentDataPointLength = self.getDataPointLength(self.currentDataPoint)
                self.dataPointPosition = 0
            for j, att in enumerate(self.attributes):
                if self.isSequence[att]:  # Need to limit the sequence to the end of the batch...
                    # Put the sequence attributes in their proper positions in the tensor array
                    # These are numeric encoding schemes.
                    attData = self.currentDataPoint[att][
                        self.dataPointPosition]  # Get the next entry in the attribute sequence for the current data point
                else:
                    # Put the context attributes in their proper positions in the tensor array
                    # These are a one-hot encoding schemes except in the case of "age" and the like
                    if self.isNominal[att]:  # Checks whether the data is nominal
                        attData = self.oneHot(self.currentDataPoint, att)  # returns a list
                    else:
                        attData = self.currentDataPoint  # Handles ordinal and numeric data

                scaledAttData = self.scaleData(attData, att)  # Rescales data if needed
                if self.isList(scaledAttData):
                    dataList.extend(scaledAttData)
                else:
                    dataList.append(scaledAttData)
            if len(dataList) == self.dataDim:
                dataBatch[i, :] = dataList
            else:
                print("Data list length: " + dataList)
                print("Data schema length: " + self.dataDim)
                raise (ValueError("Data is not formatted according to the schema"))

            self.dataPointPosition = self.dataPointPosition + 1

        return dataBatch

    """Should use three levels of grouping for data: the epoch, the batch, and the element,
    where the element is equivalent to an individual exercise routine (or an individual review, etc.)
    After each element, the state of the rnn should be reset so as to properly separate them.
    I am not sure yet how to best initialize the RNN state..."""

    def endoIterator(self, batch_size, num_steps, dataSetType):
        """This function gets called as an iterator and returns a new batch of data where each data point is an (x,y) tuple
        where the x is the inputs and the y is the labels.
        A batch is returned as an numpy array (a 3-tensor) of batch_size by num_steps by self.dataDim
        where each row is length num_steps and is shifted over by one element.
        There are batch_size number of rows."""

        # NEED TO CLEAN THIS UP AND REFERENCE THE EXAMPLE

        # This should work except for data type/size problems

        # raw_data = np.array(raw_data, dtype=np.int32)

        data_len = self.numDataPoints
        batch_len = data_len // batch_size
        epoch_size = (batch_len - 1) // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        # For these guys, the labels are simply the next sequence. This is to train the model to reprodue the text.
        # Since I am not really trying to do this, I should generate the labels seperately.
        # However, I might find that training the net this way (to predict the sequence) and then transplanting the weights into the full model might be useful...
        """for i in range(epoch_size):
            batchData=self.nextBatch(batch_size)
            data = np.zeros([batch_size, batch_len, self.dataDim])
            for j in range(batch_size):
                data[j] = batchData[batch_len * j:batch_len * (j + 1)]
            x = data[:, i*num_steps:(i+1)*num_steps]
            y = data[:, i*num_steps+1:(i+1)*num_steps+1]
            yield (x, y)"""

        # The code below is not ideal because it trains everything in order whereby each batch is comprised sequentially of the data
        # It should be OK for basic testing, however.
        # and it may miss some transitions (not sure)
        # print( epoch_size)
        for i in range(epoch_size):
            batchData = self.nextBatch(batch_size * (num_steps + 1))
            # print(batchData.shape)
            data = np.zeros([batch_size, num_steps + 1, self.dataDim])
            for j in range(batch_size):
                data[j, :, :] = batchData[(num_steps * j):((num_steps * (j + 1)) + 1), :]
            # print(data.shape)
            # x = data[:, i*num_steps:((i+1)*num_steps),:]
            x = data[:, 0:num_steps, :]
            # y = data[:, i*(num_steps+1):((i+1)*num_steps+1),:]
            y = data[:, 1:(num_steps + 1), :]
            yield (x, y)

    def scaleData(self, data, att):
        # This function provides optional rescaling of the data for optimal neural network performance.
        # It can either be run online or offline w/ results stored in a preprocessed data file (more effecient)
        if att == "speed":
            scaledData = data
            return scaledData
        elif att == "heart_rate":
            scaledData = data / 250.0  # This will be replaced with an auto-ranging version
            return scaledData
        elif att == "altitude":
            scaledData = float(data) / 10000.0  # This will be replaced with an auto-ranging version
            return scaledData
        else:
            return data

    """Might also want to define a function that returns a batch where
    each non-sequential attribute is copied so as to be as long as the longest sequential attribute in the attribute list."""

    """Write a function that returns data (a batch of data) in the format defined by the following placeholder tensors:
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        Might also be ok with having a third placeholder:
        self._context = tf.placeholder(tf.int32, [batch_size])

        Might also want to be able to return a generator that iterates (enumerates) on a batch data point by data point.

        Alternatively, I could have a generator that iterates on a batch and then another generator that iterates
        on the individual elements of a particular data point (sequence and context)"""

    def newEpoch(self):
        # A convenience function for reseting the data loader to start a new epoch
        self.createGenerator()
        self.currentDataPoint = None
        self.currentDataPointNumber = 0  # The number of data points looked at so far
        self.dataPointPosition = 0  # The position within a data point (within an exercise)
        self.trainingOrder = self.randomizeDataOrder(self.trainingSet)
        self.validationOrder = self.randomizeDataOrder(self.validationSet)
        self.testOrder = self.randomizeDataOrder(self.testSet)


class metaDataEndomondo(object):
    # For disk storage of metadata
    # Meant to be pickled and unpickled
    def __init__(self, numDataPoints, encodingLengths, oneHotEncoders, isSequence, isNominal):
        self.numDataPoints = numDataPoints
        self.encodingLengths = encodingLengths
        self.oneHotEncoders = oneHotEncoders
        # self.dataDim=dataDim
        self.isSequence = isSequence
        self.isNominal = isNominal