import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

#Might need to pass a session or add some tensorflow related imports

class inputManager(object):
    def __init__(self, inputIndicesDict, inputOrderNames, numInitialInputs=1):
        self.inputOrder = inputOrderNames
        self.inputIndices = self.convertNamesToPositions(inputIndicesDict, inputOrderNames) # A list of tuples that contains the start and end indices of each input variable
        #self.inputOrder = convertNamesToPositions(inputOrderNames) # A list where the 0th position contains the index of the first input to add, the 1st position contains the index of the second input to add, etc.
        self.inputState = np.zerosLike(self.inputOrder)#A numpy array of ones and zeros that keeps track of which inputs variables are active
        self.numActiveInputs = 0
        self.mask = self.initialMask(self.inputIndices)#initialize the mask. This should be a tensor of the same shape as the input tensor x
        
        while numInitialInputs>0: #Add the initial inputs
            self.addInput()
            numInitialInputs -= 1

    def dropin(self, x, name=None):

        """Masks a subset of a tensor based on a vector of positions and a vector that specified whether or not to mask.
        This is used to allow for a model to be trained progressively by adding input components progressively as the moidel 
        trained on the previous variables begins to converge. This will encourage the model to build a more modular representation
        based on knowledge of the input structure contained in the variableSplit paramater, which is intended to reflect seperate
        data sources or input types that have been mashed together into a single input tensor for model structure reasons.

        It is an important component of successfully trainign a contextual RNN


        Might want to write an additional function that manages the activeVars variable so that it can progressively add a variable whenever
        it gets a signal to do so. This would take in an ordering of the variables to add and possibly the current activeVars 
        (if it is not part of an object) in whihc case it would simply need to take in an ordering and return the next activeVars state.
        This later possibility might be much easier to do...


        I likely just need to make a tensor of ones and zeros that corresponds to the activeVars and is in the same shape as x.
        Then all I need to do is multiply x by this tensor.
        Could also do a bit-masking thing, but multiplication seems easier and might even be faster given the floating point gpu hardware...

      Args:
        x: A tensor.
        variableSplit: A length (numInputVars) list of tuples where each tuple corresponds to the start and end indices of the nth input variable
        inputState: A 1-D tensor of length (numInputVars) containing a 1 if the variable should be used and a zero if it should not
        name: A name for this operation (optional).
      Returns:
        A Tensor of the same shape of `x`.
      """
        #variableSplit: A 1-D tensor of length (numInputVars) containing the indices (in the input vector) at which each input variable begins

        with ops.name_scope(name, "dropin", [x]) as name:
            x = ops.convert_to_tensor(x, name="x")
            

        


        #ret = math_ops.div(x, keep_prob) * binary_tensor
        #ret.set_shape(x.get_shape())
        return tf.mul(x, self.mask) #Hopefully this is the elementwise multiplication operator. If not, we will see soon enough.

    """def addInput():
        inputToAdd = self.inputOrder(self.numActiveInputs)
        self.inputState[inputToAdd]=1
        self.numActiveInputs = self.numActiveInputs+1"""
    
    def addInput(self):
        self.inputState[self.numActiveInputs] = 1 #Update the input state
        self.numActiveInputs += 1 #Update the number of active inputs
                
        #Update the mask
        for i, attIndices in enumerate(self.inputIndices):
            if self.inputState[i]==1: #Set the elements of x corresponding to the added variable to 1
                for j in range(attIndices[0],attIndices[1]):
                    self.mask[j]=1
            elif self.inputState[i]==0:
                for j in range(attIndices[0],attIndices[1]):
                    self.mask[j]=0
                    
        
    def convertNamesToPositions(self, inputIndicesDict, inputOrderNames):
        #Converts the dictionary of attribrute indices to a list of attribute indices and puts them in order
        inputIndicesList = []
        for name in inputOrderNames:
            inputIndicesList.append(inputIndicesDict[name])
            
        return inputIndicesList
        
        
    def initialMask(self, inputIndices):
        #This function returns a mask tensor of the correct shape by examining the inputIndices and determining the length of the input vector
        
        maxIndex=0
        for i, indexPair in enumerate(inputIndices):
            if indexPair[1]>maxIndex:
                maxIndex = indexPair[1]
        
        #Create a 1-d tensorflow tensor of zeros of length maxIndex
        mask = tf.zeros([maxIndex])
        
        return mask