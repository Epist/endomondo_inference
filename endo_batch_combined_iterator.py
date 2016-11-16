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

    def endoIteratorSupervised(self, batch_size, num_steps, trainValidTest, targetAtt):
            #Does the same thing as the endoIterator, except for a model with separate supervised targets (targets are not the next element in the sequence)

            batchGen = self.batchIteratorSupervised(1, trainValidTest, targetAtt)

            data_len = self.numDataPoints
            batch_len = data_len // batch_size
            epoch_size = (batch_len - 1) // num_steps

            inputDataDim = self.getInputDim(targetAtt)
            targetDataDim = self.getTargetDim(targetAtt)

            if epoch_size == 0:
                raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

            nextRow=[batchGen.next() for x in range(num_steps)]#Fill the first row (with both inputs and targets)
            for i in range(epoch_size):
                #[batchInputs, batchTargets] = batchGen.next()
                inputData = np.zeros([batch_size*num_steps, num_steps, inputDataDim])
                targetData = np.zeros([batch_size*num_steps, num_steps, targetDataDim])
                for j in range(batch_size):
                    for k in range(num_steps):
                        print(" j: " + str(j) + " k: " + str(k))
                        currentRow=nextRow
                        #print(inputData[((j*num_steps)+k), :, :])
                        inputData[((j*num_steps)+k), :, :]  = [currentRow[x][0] for x in range(num_steps)]
                        targetData[((j*num_steps)+k), :, :] = [currentRow[x][1] for x in range(num_steps)]
                        nextRow[0:num_steps-1] = currentRow[1:num_steps]
                        nextRow[num_steps-1]=batchGen.next()
                        #inputData[(j*num_steps)+k, :, :] = batchInputs[((num_steps * j)+k):((num_steps * (j + 1))+k), :]
                        #targetData[(j*num_steps)+k, :, :] = batchTargets[((num_steps * j)+k):((num_steps * (j + 1))+k), :]
                #print("Input size: "+str(np.size(batchInputs)))
                #print("Output size: "+str(np.size(inputData)))
                #x = data[:, i*num_steps:((i+1)*num_steps),:]
                x = inputData
                #y = data[:, i*(num_steps+1):((i+1)*num_steps+1),:]
                y = targetData
                yield (x, y)