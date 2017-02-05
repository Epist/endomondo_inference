"""A Contextual LSTM for prediction on the Endomondo Exercise Dataset


The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size


To run:

$ python Endomondo_RNN_tests.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime

import numpy as np
import tensorflow as tf
import pickle
import sys, argparse

# from tensorflow.models.rnn.ptb import reader
# from tensorflow.models.rnn import *
#from dataInterpreter_Endomondo_fixedInputs import dataInterpreter, metaDataEndomondo
from data_interpreter_with_excision import dataInterpreter, metaDataEndomondo
from inputManager import inputManager

# flags = tf.flags
logging = tf.logging

# flags.DEFINE_string(
#    "model", "small",
#    "A type of model. Possible options are: small, medium, large.")
# flags.DEFINE_string("data_path", None, "data_path")

# FLAGS = flags.FLAGS

global model
model = "Larry"
#model = "FixedDropin"
data_path = "../multimodalDBM/endomondoHR_proper.json"
summaries_dir = "logs"
# endoFeatures = ["speed", "sport", "heart_rate", "gender", "altitude"]#The features we want the model to care about
global endoFeatures
endoFeatures = ["sport", "heart_rate", "gender", "altitude", "time_elapsed", "distance", "new_workout", "derived_speed", "userId"]
#inputOrderNames = ["new_workout", "derived_speed", "sport", "altitude", "gender", "distance", "time_elapsed", "userId"] #add "userId"

numInitialInputs = 1
#endoFeatures = ["sport", "heart_rate", "gender", "altitude"]
trainValTestSplit = [0.8, 0.1, 0.1]
targetAtt = "heart_rate"
global lossType
lossType = "RMSE" #MAE, RMSE
savePredictions = True #Save prediction and/or input sequences for later viewing of the model-data relationship
modelRunIdentifier=datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
maxLoggingSteps = 1000
interDropinInterval = 5
zMultiple = 5
global dropInEnabled
dropInEnabled = False
global fnEnd
fnEnd = ''

global inputOrderNames
inputOrderNames = [x for x in endoFeatures if x!=targetAtt]
#print(inputOrderNames)

class EndoModel(object):
    """The Endomondo Contextual LSTM model."""

    def __init__(self, is_training, config, dropinManager):
        self.is_training=is_training
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.num_layers = config.num_layers
        size = config.hidden_size
        # vocab_size = config.vocab_size
        #dataDim = config.dataDim
        inputShape = config.inputShape
        targetShape = config.targetShape
        pos_weight = config.pos_weight  # This is a coefficient that weights the relative importance of positive prediction error and negative prediction error. The default is 1 (equal weight.)

        self._input_data = tf.placeholder(tf.float32, [batch_size*num_steps, inputShape])
        self._targets = tf.placeholder(tf.float32, [batch_size*num_steps, targetShape])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        # Other resonable activation functions include: activation=tf.nn.relu and activation=tf.nn.softmax
        # i.e. lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, activation=tf.nn.relu)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # with tf.device("/cpu:0"):
        #  embedding = tf.get_variable("embedding", [vocab_size, size])
        #  inputs = tf.nn.embedding_lookup(embedding, self._input_data)
        inputs = self._input_data

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        #inputs = [tf.squeeze(input_, [1])
        #          for input_ in tf.split(0, num_steps, inputs)]
        inputs_to_save = inputs = [input_ for input_ in tf.split(0, num_steps, inputs)]
        #print(tf.get_shape(inputs))
        if dropInEnabled:
            inputs = dropinManager.dropin(inputs)#Drop in component
        #outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

        # Might need to change this stuff...
        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.Variable(tf.ones([size, targetShape]), trainable=False)
        #softmax_b = tf.Variable(tf.ones([targetShape]), trainable=False)
        #logits = tf.matmul(output, softmax_w) + softmax_b  # Probably need to change this...
        logits = tf.matmul(output, softmax_w)
        
        variable_summaries(inputs, 'inputs')
        variable_summaries(logits, 'logits')

        # Need a new loss function here...
        #loss = tf.nn.weighted_cross_entropy_with_logits(
        #    [logits],
        #    tf.reshape(self._targets, [-1, batch_size * num_steps, targetShape]),
        #    pos_weight)
        
        #reshapedTargets=tf.reshape(self._targets, [-1, batch_size * num_steps, targetShape])
        reshapedTargets=tf.reshape(self._targets, [-1, targetShape])
        #print(tf.get_shape(reshapedTargets))
        #print(tf.get_shape(logits))
        logTarDiff = tf.sub(reshapedTargets, logits)
        if lossType=="RMSE":#Root mean squared error
            loss = tf.square(logTarDiff)
        elif lossType=="MAE":#Mean absolute error
            #loss = tf.reduce_mean(tf.abs(logTarDiff))
            loss = tf.abs(logTarDiff)
        else:
            raise(Exception("Must specify a loss function"))
            
        variable_summaries(loss, 'loss')
        variable_summaries(reshapedTargets, 'targets')
        variable_summaries(logTarDiff, 'logit-target difference')
        # loss = tf.nn.softmax_cross_entropy_with_logits(
        #    [logits],
        #    [tf.reshape(self._targets, [-1])],
        #    [tf.ones([batch_size * num_steps])])

        # loss = tf.nn.seq2seq.sequence_loss_by_example(
        #    [logits],
        #    [tf.reshape(self._targets, [-1])],
        #    [tf.ones([batch_size * num_steps])])
        self.inputs = inputs_to_save
        self.logits = logits
        self.reshapedTargets = reshapedTargets
        self.outputs=outputs
        self.output=output
        
        self.loss = loss
        if lossType=="RMSE":
            self._cost = cost = tf.sqrt(tf.reduce_sum(loss)) / batch_size
        elif lossType=="MAE":
            self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        
        self.merged = tf.merge_all_summaries()

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        #print(tvars)
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class ReallySmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 40
    hidden_size = 100
    max_epoch = 2
    max_max_epoch = 8
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    # vocab_size = 10000
    dataDim = 0
    inputShape = []
    targetShape = []
    pos_weight = 1


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    # vocab_size = 10000
    dataDim = 0
    inputShape = []
    targetShape = []
    pos_weight = 1


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    # vocab_size = 10000
    dataDim = 0
    inputShape = []
    targetShape = []
    pos_weight = 1


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    # vocab_size = 10000
    dataDim = 0
    inputShape = []
    targetShape = []
    pos_weight = 1


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    # vocab_size = 10000
    dataDim = 0
    inputShape = []
    targetShape = []
    pos_weight = 1


class LarryConfig(object):
    """Larry's custom config"""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 700
    max_epoch = 12
    max_max_epoch = 55
    keep_prob = 0.4
    lr_decay = 0.85
    batch_size = 20
    # vocab_size = 10000
    dataDim = 0
    inputShape = []
    targetShape = []
    pos_weight = 1

class fixedDropInConfig(object):
    """Config for dropin with a fixed interval between adding variables"""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = interDropinInterval
    max_max_epoch = None #Set in runtime
    keep_prob = 0.5
    lr_decay = 0.95
    batch_size = 20
    # vocab_size = 10000
    dataDim = 0
    inputShape = []
    targetShape = []
    pos_weight = 1
    
    
def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    #for i, val in enumerate(tf.reshape(var, [-1]).eval()):
    #    tf.scalar_summary('full/' + name + "_index:" + str(i), val)
    tf.histogram_summary(name, var)

def run_epoch(session, m, data_interp, eval_op, trainValidTest, epochNum, verbose=False, writer=None):
    """Runs the model on the given data."""
    epoch_size = ((data_interp.numDataPoints // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    
    #if (epochNum%10==0) or (epochNum%10==1):
    #    writeEpoch=1
    #else:
    #    writeEpoch=0
    writeEpoch=1     

    # c and h are the two components of the lstm state tuple
    # See https://www.tensorflow.org/versions/r0.9/api_docs/python/rnn_cell.html#classes-storing-split-rnncell-state
    # Must handle the seperate lstm states seperately since the multiRNN class doesn't yet have a way to do this for tuple states...
    
    if m.num_layers==2:
        state1_c = m.initial_state[0].c.eval()
        state1_h = m.initial_state[0].h.eval()
        state2_c = m.initial_state[1].c.eval()
        state2_h = m.initial_state[1].h.eval()
    elif m.num_layers==1:
        state1_c = m.initial_state[0].c.eval()
        state1_h = m.initial_state[0].h.eval()

    #state1 = (state1_c, state1_h)  # the initial state of the first lstm
    #state2 = (state2_c, state2_h)  # the initial state of the second lstm

    # data_interp.newEpoch()
    dataGen = data_interp.endoIteratorSupervised(m.batch_size, m.num_steps, trainValidTest)  # A generator over the endomondo data
    # global dataGenTest
    # dataGenTest = dataGen
    # global modelTest
    # modelTest=data_interp
    
    targetSeq=[]
    logitSeq=[]
    inputSeq=[]
    outputsSeq=[]
    outputSeq=[]
    
    for step, (x, y) in enumerate(dataGen):
        
        if m.num_layers==2:
            feed_dictionary = {m.input_data: x, m.targets: y,
                               m.initial_state[0].c: state1_c,
                               m.initial_state[0].h: state1_h,
                               m.initial_state[1].c: state2_c,
                               m.initial_state[1].h: state2_h,
                               }
        elif m.num_layers==1:
            feed_dictionary = {m.input_data: x, m.targets: y,
                               m.initial_state[0].c: state1_c,
                               m.initial_state[0].h: state1_h,
                               }

        # feed_dict.update( network.all_drop )
        
        #if True:
        if m.is_training:
            if m.num_layers==2:
                loss, cost, state1_c, state1_h, state2_c, state2_h, summary, _ = session.run([m.loss, m.cost,
                                                                                     m.final_state[0].c,
                                                                                     m.final_state[0].h,
                                                                                     m.final_state[1].c,
                                                                                     m.final_state[1].h,
                                                                                     m.merged,
                                                                                     eval_op],
                                                                                     feed_dict=feed_dictionary)
            elif m.num_layers==1:
                loss, cost, state1_c, state1_h, summary, _ = session.run([m.loss, m.cost,
                                                                                     m.final_state[0].c,
                                                                                     m.final_state[0].h,
                                                                                     m.merged,
                                                                                     eval_op],
                                                                                     feed_dict=feed_dictionary)
        
            writer.add_summary(summary, step)
        else:
            if m.num_layers==2:
                loss, cost, state1_c, state1_h, state2_c, state2_h, targets, logits, inputs, _ = session.run([m.loss, m.cost,
                                                                                                         m.final_state[0].c,
                                                                                                         m.final_state[0].h,
                                                                                                         m.final_state[1].c,
                                                                                                         m.final_state[1].h,
                                                                                                         m.reshapedTargets,
                                                                                                         m.logits,
                                                                                                         m.inputs,
                                                                                                         eval_op],
                                                                                                         feed_dict=feed_dictionary)
            elif m.num_layers==1:
                loss, cost, state1_c, state1_h, targets, logits, inputs, _ = session.run([m.loss, m.cost,
                                                                                                         m.final_state[0].c,
                                                                                                         m.final_state[0].h,
                                                                                                         m.reshapedTargets,
                                                                                                         m.logits,
                                                                                                         m.inputs,
                                                                                                         eval_op],
                                                                                                         feed_dict=feed_dictionary)
                
            if (step<maxLoggingSteps)&(savePredictions is True)&(writeEpoch==1):
                targetSeq.extend(targets)
                logitSeq.extend(logits)
                #print(inputs[0])
                for inputStep in inputs:
                    inputSeq.extend(data_interp.dataDecoder(inputStep))
                #inputSeq.extend(data_interp.dataDecoder(inputs[0]))
                #inputSeq.extend(inputs)
                #outputsSeq.extend(outputs)
                #outputSeq.extend(output)

        #state1 = (state1_c, state1_h)
        #state2 = (state2_c, state2_h)
        
        

        # print(cost)

        costs += cost
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            #print("Step: " + str(step))
            # print(np.log(-costs//iters))
            print("%.3f %s: %.6f speed: %.0f dpps" %
                  (step * 1.0 / epoch_size, lossType, (costs / iters),
                   iters * m.batch_size / (time.time() - start_time)))
    
    if (m.is_training is False) & (savePredictions is True) & (writeEpoch==1):
        print("Saving data to file")
        saveData(targetSeq, logitSeq, inputSeq, outputsSeq, outputSeq, epochNum)
    
    # print(costs)
    # print(iters)
    return (costs / iters)

def saveData(targetSeq, logitSeq, inputSeq, outputsSeq, outputSeq, epochNum):
    global fnEnd
    fileName= "logs/fullData/" + modelRunIdentifier + "_epoch_" + str(epochNum+1) + fnEnd
    dataContents = dataEpoch(targetSeq, logitSeq, inputSeq, outputsSeq, outputSeq, epochNum, )
    with open(fileName, "wb") as f:
            pickle.dump(dataContents, f)
            
class dataEpoch(object):
    def __init__(self, targetSeq, logitSeq, inputSeq, outputsSeq, outputSeq, epochNum):
        self.inputSeq = inputSeq
        self.targetSeq = targetSeq
        self.logitSeq = logitSeq
        #self.outputsSeq = outputsSeq
        #self.outputSeq = outputSeq
        global endoFeatures
        global inputOrderNames
        global model
        global lossType
        self.epochNum = epochNum
        self.endoFeatures = endoFeatures
        self.inputOrderNames = inputOrderNames
        self.targetAtt = targetAtt
        self.modelType = model
        self.lossType = lossType
        self.trainValTestSplit = trainValTestSplit
        self.modelRunIdentifier = modelRunIdentifier
        self.zMultiple = zMultiple

def get_config():
    if model == "small":
        return SmallConfig()
    elif model == "medium":
        return MediumConfig()
    elif model == "large":
        return LargeConfig()
    elif model == "test":
        return TestConfig()
    elif model == "Larry":
        return LarryConfig()
    elif model == "really small":
        return ReallySmallConfig()
    elif model == "FixedDropin":
        config = fixedDropInConfig()
        config.max_max_epoch = ((len(inputOrderNames)-numInitialInputs)+2)*interDropinInterval
        return config
    else:
        raise ValueError("Invalid model: %s", model)

def parse_args(argv):
    
    #if there is an argument specified for error metric, attributes, or drop in, overwrite the defaults for the corresponding variables to the values specified in the arguments
    #also print the new values and the fact that they were changed for confirmation
    
    parser = argparse.ArgumentParser(description='Specify some model parameters')
    
    parser.add_argument('-di', dest='drop_in', action='store_true', default=False, help='Use dropin')
    parser.add_argument('-em', dest='lossType', action='store', help='Specify the error metric')
    parser.add_argument('-a', dest='attributes', action='store', nargs='+', help='Specify the attributes')
    parser.add_argument('-fn', dest='fileNameEnding', action='store', help='Append an identifying string to the end of output files')
    
    
    args = parser.parse_args()
    
    print(args)
    
    if args.drop_in:
        global dropInEnabled
        dropInEnabled = True
        global model
        model = "FixedDropin"
        print("Toggled drop in from command line")
        
    #try:
    if args.lossType == "RMSE" or args.lossType == "MAE":
        global lossType
        lossType = args.lossType
        print("Added loss type " + args.lossType + " from command line")
    else:
        print(args.lossType + " is not a valid loss type. Reverting to " + lossType)
    #except:
    #    pass
    
    #try:
    if args.attributes is not None:
        global endoFeatures
        endoFeatures = args.attributes
        global inputOrderNames
        inputOrderNames = [x for x in endoFeatures if x!=targetAtt]
        print("Added attributes from command line: " + str(endoFeatures))
    
    if args.fileNameEnding is not None:
        global fnEnd
        fnEnd = args.fileNameEnding
        

def main(argv):
    if not data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    parse_args(argv)
    
    # raw_data = reader.ptb_raw_data(data_path)
    # train_data, valid_data, test_data, _ = raw_data
    endo_reader = dataInterpreter(fn=data_path, scaleVals=True)
    endo_reader.buildDataSchema(endoFeatures, targetAtt, trainValTestSplit, zMultiple)
    
    inputIndicesDict=endo_reader.inputIndices

    inputShape = endo_reader.getInputDim(targetAtt)
    targetShape = endo_reader.getTargetDim(targetAtt)

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 50 #Was originally set to 1
    #config.dataDim = dataShape
    config.inputShape = inputShape
    config.targetShape = targetShape
    #eval_config.dataDim = dataShape
    eval_config.inputShape = inputShape
    eval_config.targetShape = targetShape
    
    if dropInEnabled:
        print("Input order: " + str(inputOrderNames))

    with tf.Graph().as_default(), tf.Session() as session:
        #with tf.device('/gpu:1'):
        dropinManager = inputManager(inputIndicesDict, inputOrderNames, numInitialInputs)

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = EndoModel(is_training=True, config=config, dropinManager=dropinManager)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = EndoModel(is_training=False, config=config, dropinManager=dropinManager)
            mtest = EndoModel(is_training=False, config=eval_config, dropinManager=dropinManager)

        train_writer = tf.train.SummaryWriter(summaries_dir + '/train', session.graph)
        test_writer = tf.train.SummaryWriter(summaries_dir + '/test')

        tf.initialize_all_variables().run()
        if dropInEnabled:
            print("Starting with " + str(dropinManager.numActiveInputs) + " input(s)")
        for i in range(config.max_max_epoch):
            if dropInEnabled:
                if i%interDropinInterval==0:
                    #Simple drop in condition
                    addedInput = dropinManager.addInput()
                    print("Adding input number " + str(dropinManager.numActiveInputs) + " : " + addedInput)
            epochNum=i+1
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, endo_reader, m.train_op, 'train', epochNum, 
                                         verbose=True, writer=train_writer)
            print("Epoch: %d Train %s: %.6f" % (i + 1, lossType, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid, endo_reader, tf.no_op(), 'valid', epochNum, verbose=True, writer=test_writer)
            print("Epoch: %d Valid %s: %.6f" % (i + 1, lossType, valid_perplexity))

        test_perplexity = run_epoch(session, mtest, endo_reader, tf.no_op(), 'test', epochNum+1, writer=test_writer)
        print("Test %s: %.4f" % (lossType, test_perplexity))

if __name__ == "__main__":
    main(sys.argv)


