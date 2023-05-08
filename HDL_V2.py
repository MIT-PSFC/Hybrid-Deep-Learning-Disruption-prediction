# -*- coding: utf-8 -*-
"""

@author: jinxiang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from sklearn import metrics
#from absl import flags
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import scipy.io as sio
import os
import time 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
#import distribution_strategy_utils as ds_utils
'''
FLAGS = flags.FLAGS
import sys 
FLAGS(sys.argv)

flags.DEFINE_string(
    'export_dir', '.\\saved_model',
    'Directory of exported SavedModel.')
flags.DEFINE_integer(
    'epochs',    1,
    'Number of epochs to train.')
flags.DEFINE_integer(
    'seqlength', 10,
    'Number of epochs to train.')
flags.DEFINE_integer(
    'numFeatures', 12,
    'Number of features.')
flags.DEFINE_bool(
    'use_keras_save_api', False,
    'Uses tf.keras.models.save_model() on the feature extractor '
    'instead of tf.saved_model.save() on a manually wrapped version. '
    'With this, the exported model as no hparams.')
flags.DEFINE_bool(
    'export_print_hparams', False,
    'If true, the exported function will print its effective hparams.')
'''
SIGNAL_INPUT_NAME = 'signal'
LABEL_INPUT_NAME = 'label'

#Define MSTConv layer. Unit is the number of kernel for each 1D conv layer (default value: 10)
class attentionblock(layers.Layer):
    def __init__(self, regularizer, unit=10, name=''):
        super(attentionblock, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv1D(unit, 1, padding='SAME', 
                                            kernel_regularizer=regularizer())
        self.conv2 = tf.keras.layers.Conv1D(unit, 2, padding='SAME', 
                                            kernel_regularizer=regularizer())
        self.conv3 = tf.keras.layers.Conv1D(unit, 3, padding='SAME', 
                                            kernel_regularizer=regularizer())
        self.conv4 = tf.keras.layers.Conv1D(unit, 4, padding='SAME', 
                                            kernel_regularizer=regularizer())
        self.conv5 = tf.keras.layers.Conv1D(unit, 5, padding='SAME', 
                                            kernel_regularizer=regularizer())
        self.conv6 = tf.keras.layers.Conv1D(unit, 6, padding='SAME', 
                                            kernel_regularizer=regularizer())
        self.bn1 = tf.keras.layers.BatchNormalization(beta_regularizer=regularizer(), 
                                                       gamma_regularizer=regularizer())
    
    def call(self, input_tensor, training=False):
        x = tf.concat([self.conv1(input_tensor), self.conv2(input_tensor), 
                       self.conv3(input_tensor), self.conv4(input_tensor),
                       self.conv5(input_tensor), self.conv6(input_tensor)], -1)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        return x

def auc(x, y):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule.
    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.  For an alternative
    way to summarize a precision-recall curve, see
    :func:`average_precision_score`.
    Parameters
    ----------
    x : ndarray of shape (n,)
        x coordinates. These must be either monotonic increasing or monotonic
        decreasing.
    y : ndarray of shape, (n,)
        y coordinates.
    Returns
    -------
    auc : float
    See Also
    --------
    roc_auc_score : Compute the area under the ROC curve.
    average_precision_score : Compute average precision from prediction scores.
    precision_recall_curve : Compute precision-recall pairs for different
        probability thresholds.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr)
    0.75
    """
    

    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)

    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing "
                             ": {}.".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in np.trapz do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area

#Build tensorflow input pipeline using saved .mat dataset
def make_dataset(path,batch_size=200):
    
    xdata_path = path
    #load training data file
    x = sio.loadmat(xdata_path) 
    #training dataset
    xdata = x['data'].astype(np.float32)
    #training label
    ydata = x['trainingClasses']
    #reshape dataset to num_samp*seq_length*num_features
    xdata = xdata.reshape(-1, 10, 14)
    ydata = ydata.reshape(-1,)
    #reshuffle training data
    l = len(xdata)
    num_step=l//batch_size
    indices = np.arange(l)
    np.random.shuffle(indices)
    xdata = xdata[indices]
    ydata = ydata[indices]

    #label smoothing: correction0 is the soft label smoothing to non-disruptive class
    # correction1 is the soft label smoothing to disruptive class
    index0 = np.where(ydata==0)[0]
    correction0 = np.asarray([-0.00, 0.00])
    index1 = np.where(ydata==1)[0]
    correction1 = np.asarray([0.00, -0.00])
    ydata = tf.one_hot(ydata, 2).numpy()
    ydata[index0]=ydata[index0]+correction0
    ydata[index1]=ydata[index1]+correction1

    #convert training data to tensorflow input pipeline, default shuffle size is 10000
    dataset = tf.data.Dataset.from_tensor_slices((xdata, ydata))
    dataset = dataset.cache().shuffle(10000).batch(batch_size, drop_remainder=True).repeat()
    return dataset, num_step

#The HDL_V2 model, the two parameters are l2_regularization strength and dropout rate
class make_predictor(tf.keras.Model):
    def __init__(self, l2_strength, dropout_rate):
        super().__init__()
        self.regularizer = lambda: tf.keras.regularizers.l2(l2_strength)
        self.ab1 = attentionblock(self.regularizer, name='AttentionBlock_1')
        self.ab2 = attentionblock(self.regularizer, name='AttentionBlock_2')
        self.ab3 = attentionblock(self.regularizer, name='AttentionBlock_3')
        self.ab4 = attentionblock(self.regularizer, unit=15, name='AttentionBlock_4')
        self.ab5 = attentionblock(self.regularizer, unit=15, name='AttentionBlock_5')
        self.ab6 = attentionblock(self.regularizer, unit=15, name='AttentionBlock_6')
        self.ab7 = attentionblock(self.regularizer, name='AttentionBlock_7')
        '''
        self.gru1 = tf.keras.layers.GRU(130, kernel_regularizer=self.regularizer, 
                                        recurrent_regularizer=self.regularizer, 
                                        return_sequences=True, name='GRU1')
        self.gru2 = tf.keras.layers.GRU(90, kernel_regularizer=self.regularizer, 
                                        recurrent_regularizer=self.regularizer, 
                                        return_sequences=True, name='GRU2')
        '''
        self.drop1 = tf.keras.layers.Dropout(dropout_rate, name='dropout1')
        self.drop2 = tf.keras.layers.Dropout(dropout_rate, name='dropout2')
        self.drop3 = tf.keras.layers.Dropout(dropout_rate, name='dropout3')
        self.drop4 = tf.keras.layers.Dropout(dropout_rate, name='dropout4')
    
        #self.flatten2 = tf.keras.layers.Flatten()
        self.event_drop1=tf.keras.layers.Dropout(dropout_rate, name='event_drop1')
        self.event_dense1=tf.keras.layers.Dense(64, 
                              kernel_regularizer=self.regularizer(), name='event_dense1')
        self.event_dense2=tf.keras.layers.Dense(30, 
                              kernel_regularizer=self.regularizer(), name='event_dense2')
        self.event_dense3=tf.keras.layers.Dense(11, activation='sigmoid',
                              kernel_regularizer=self.regularizer(), name='event_dense3')
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(12, name='dense1',
                             kernel_regularizer=self.regularizer())
        self.bn1 = tf.keras.layers.BatchNormalization(beta_regularizer=self.regularizer(), 
                                                      gamma_regularizer=self.regularizer())
        self.bn2 = tf.keras.layers.BatchNormalization(beta_regularizer=self.regularizer(), 
                                                      gamma_regularizer=self.regularizer())
        self.bn3 = tf.keras.layers.BatchNormalization(beta_regularizer=self.regularizer(), 
                                                      gamma_regularizer=self.regularizer())

        self.classification=tf.keras.layers.Dense(2, activation='softmax',
                              kernel_regularizer=self.regularizer(), name='classification')
        
    def forward(self, x, training=False):
        x=self.ab1(x, training=training)
        xx=self.ab2(x, training=training)
        x=x+xx
        
        
        x=self.ab4(x, training=training)
        xx=self.ab5(x, training=training)
        x=x+xx
        xx=self.ab6(x, training=training)
        x=x+xx
        x=self.drop3(x, training=training)
        
        
        x=self.ab3(x, training=training)
        xx=self.ab7(x, training=training)
        x=x+xx
        x=self.flatten1(x)
        x=self.drop1(x, training=training)
        
       
        x=self.dense1(x)
        x=self.bn1(x, training=training)
        x=tf.nn.relu(x)
        x=self.drop2(x, training=training)
        x=self.classification(x)
        return x

#The loss function
@tf.function
def loss_fn(ydata, y_pred):
    disruptivity_loss=tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(ydata, y_pred))
    
    return disruptivity_loss

#Save the trained model
def wrap_keras_model_for_export(model, batch_input_shape,
                                set_hparams, default_hparams):
  """Wraps `model` for saving and loading as SavedModel."""
  # The primary input to the module is a Tensor with a batch of time series.
  # Here we determine its spec.
  inputs_spec = tf.TensorSpec(shape=batch_input_shape, dtype=tf.float32)

  # The module also accepts certain hparams as optional Tensor inputs.
  # Here, we cut all the relevant slices from `default_hparams`
  # (and don't worry if anyone accidentally modifies it later).
  if default_hparams is None: default_hparams = {}
  hparam_keys = list(default_hparams.keys())
  hparam_defaults = tuple(default_hparams.values())
  hparams_spec = {name: tf.TensorSpec.from_tensor(tf.constant(value))
                  for name, value in default_hparams.items()}

  # The goal is to save a function with this argspec...
  argspec = tf_inspect.FullArgSpec(
      args=(['inputs', 'training'] + hparam_keys),
      defaults=((False,) + hparam_defaults),
      varargs=None, varkw=None,
      kwonlyargs=[], kwonlydefaults=None,
      annotations={})
  # ...and this behavior:
  def call_fn(inputs, training, *args):
    if FLAGS.export_print_hparams:
      args = [tf.keras.backend.print_tensor(args[i], 'training=%s and %s='
                                            % (training, hparam_keys[i]))
              for i in range(len(args))]
    kwargs = dict(zip(hparam_keys, args))
    if kwargs: set_hparams(model, **kwargs)
    return model(inputs, training=training)

  # We cannot spell out `args` in def statement for call_fn, but since
  # tf.function uses tf_inspect, we can use tf_decorator to wrap it with
  # the desired argspec.
  def wrapped(*args, **kwargs):
    return call_fn(*args, **kwargs)
  traced_call_fn = tf.function(
      tf_decorator.make_decorator(call_fn, wrapped, decorator_argspec=argspec))

  # Now we need to trigger traces for all supported combinations of the
  # non-Tensor-value inputs.
  for training in (True, False):
    traced_call_fn.get_concrete_function(inputs_spec, training, **hparams_spec)

  # Finally, we assemble the object for tf.saved_model.save().
  obj = tf.train.Checkpoint()
  obj.__call__ = traced_call_fn
  obj.trainable_variables = model.trainable_variables
  obj.variables = model.trainable_variables + model.non_trainable_variables
  # Make tf.functions for the regularization terms of the loss.
  obj.regularization_losses = [_get_traced_loss(model, i)
                               for i in range(len(model.losses))]
  return obj


def _get_traced_loss(model, i):
  """Returns tf.function for model.losses[i] with a trace for zero args.
  The intended usage is
    [_get_traced_loss(model, i) for i in range(len(model.losses))]
  This is better than
    [tf.function(lambda: model.losses[i], input_signature=[]) for i ...]
  because it avoids capturing a loop index in a lambda, and removes any
  chance of deferring the trace.
  Args:
    model: a Keras Model.
    i: an integer between from 0 up to but to len(model.losses).
  """
  f = tf.function(lambda: model.losses[i])
  _ = f.get_concrete_function()
  return f

#evaluate the prediction performance of the trained HDL_ensemble
def calculate_AUC(out_list, length, testShotlist, testTime_until_disrupt, testClasses):
    Disruptivity=[]                                                                                                 
    num_test_sample=len(testClasses)
    num_P=len(np.where(testClasses==1)[0])
    num_N=len(np.where(testClasses==0)[0])
    out=np.asarray(out_list)
    out=np.mean(out, axis=0).reshape((-1,))
    length=np.asarray(length)
    sp=np.cumsum(length)
    disruptivity=np.split(out,sp,axis=0)
    Disruptivity=disruptivity[:-1] 
    
    length=np.asarray(length)
    sp=np.cumsum(length)
    
    classes=np.zeros(num_test_sample,dtype='int16')
    th=np.linspace(0.0,1.0,1001).tolist()
    TPR_list=[]
    FPR_list=[]
    for thresh in th:
        classes=np.zeros(num_test_sample,dtype='int16')
        for i in range(num_test_sample):
            index=np.where(Disruptivity[i]>thresh)
            if len(index[0])>0:
                classes[i]=1
        indexx=np.where((classes==1)&(testClasses==1))
        testTime_until_disrupt_effec=testTime_until_disrupt[indexx]
        b=np.zeros(num_test_sample)  
        for i in range(num_test_sample):
            index=np.where(Disruptivity[i]>thresh)
            if len(index[0])>0:
                b[i]=index[0][0]
        b=b[indexx]
        b=np.asarray(b,dtype='int16')
        tin=b+10-1
        time_new=[]
        for i in range(len(tin)):
            until=testTime_until_disrupt_effec[i].reshape(-1,1).tolist()
            time_new.append(until[tin[i]])
        time_new=np.asarray(time_new)
        TPR_list.append(len(np.where(time_new>0.05)[0])/num_P)
        FPR_list.append(len(np.where((classes==1)&(testClasses==0))[0])/num_N)
    fpr=np.array(FPR_list)
    tpr=np.array(TPR_list)
    AUC=auc(fpr, tpr)
    return Disruptivity, FPR_list, TPR_list, AUC



def train(model, dataset, num_step, epochs=16, lr=5e-4, log_steps=20):
    #DNN optimizer
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=15)
    iterator = iter(dataset)    
    xdata, ydata = iterator.get_next()
    y_pred=model.forward(xdata)
    #train the HDL model
    for e in range(epochs):
        total_loss=0
        start = time.time()
        for i in range(num_step):
            xdata, ydata = iterator.get_next()
            #get the gradient of the trainable parameters based on training loss
            with tf.GradientTape() as tape:
                y_pred=model.forward(xdata, training=True)
                train_loss=loss_fn(ydata, y_pred)
            gradients = tape.gradient(train_loss, model.trainable_variables)
            #adjust the trainable parameters
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += train_loss.numpy()
            loss = total_loss / (i+1.0)
            if ((i+1) % log_steps) ==0:
                print('Epoch {}, Step {}: loss = {}, Time to run {} steps = {}'.format(e+1, i+1, loss, log_steps, time.time() - start))
                start = time.time()
    
    return model

#get the output disruptivity from the trained model and test data           
def batch_test(model, x, batch_size):
    data_size = x.shape[0]
    if (data_size%batch_size) == 0:
        num_test=data_size//batch_size
    else:
        num_test=(data_size//batch_size)+1
    results=[]
    for i in range(num_test):
        if i<data_size//batch_size:
            data=x[i*batch_size:(i+1)*batch_size]
        else:
            data=x[i*batch_size:]
        y_pred = model.forward(data)
        y_pred=y_pred.numpy()
        
        results.append(y_pred)
        
    
    results=np.concatenate(results)
    return results


def main():
    #load test set
    box = sio.loadmat('./C_Mod_sim_SPARC_HP_test.mat')
    testDataset = box['testDataset'].ravel()
    testShotlist = box['testShotlist'].ravel()
    testTime_until_disrupt = box['testTime_until_disrupt'].ravel()
    testClasses = box['testClasses'].ravel()
    Numtest = len(testShotlist)
    test=[]
    length=[]
    #number of HDL model in an ensemble, default value is 12
    num_network = 12
    #number of independent test for each combo of hyperparameters
    Num_trial = 3
    
    #preprocess the test dataset and convert it to the format of the model input data
    for k in range(Numtest):
          l = np.shape(testDataset[k])[0]
          shotdata = testDataset[k]
          inputdata = [shotdata[ j:j+10,:].reshape(1,10,14) for j in range(0, l-10+1)]
          length.append(len(inputdata))
          test=test+inputdata
    dtest = np.concatenate(test)
    AUC_list=[]
    Disruptivity_list=[]
    TPR_list_list=[]
    FPR_list_list=[]
    predictor_list=[]
    
    #hparam search: batch_size, epoch_num and learning rate are three key hyperparams
    batch_list=[192, 128, 96]
    epoch_list=[8, 16, 24]
    lr_list=[1e-4, 5e-4, 1e-3]
    hyperparams=[]
    for i in batch_list:
        for j in epoch_list:
            for l in lr_list:
                hyperparams.append((i,j,l))
    l2_strength = 1e-3
   
    for param in hyperparams:
        AUC=0
        #load training set
        dataset, num_step = make_dataset('./C_Mod_sim_SPARC_HP_training.mat',param[0])
        for _ in range(Num_trial):
            #predictor_list=[]
            out_list=[]
            for _ in range(num_network):
                
                classifier = make_predictor(l2_strength=1e-3, dropout_rate=0.1)
                classifier = train(classifier, dataset, num_step, 
                                   epochs=param[1], lr=param[2])
                
                output = batch_test(classifier, dtest, 1000)
                del classifier
                #predictor_list.append(model)
                output=output[:,1]
                out_list.append(output)
            Disruptivity, FPR_list, TPR_list, AUC_temp=calculate_AUC(out_list, length, testShotlist, 
                                                                     testTime_until_disrupt, testClasses)
            predictor_list.append(AUC_temp)
            print(AUC_temp)
            Disruptivity_list.append(Disruptivity)
            TPR_list_list.append(TPR_list)
            FPR_list_list.append(FPR_list)
            AUC=AUC+AUC_temp
        del dataset
        AUC=AUC/Num_trial
        print(AUC)
        AUC_list.append(AUC)
        #save intermediate results
        sio.savemat('./C_Mod_sim_SPARC_train_C_Mod_HP.mat', {'AUC': predictor_list,'disruptivity': 
            Disruptivity_list,'TPR_list': TPR_list_list,'FPR_list': FPR_list_list,'AUC_ave': AUC_list})
    
    return AUC_list, Disruptivity_list, TPR_list_list, FPR_list_list, predictor_list



if __name__ == '__main__':
  AUC_list, Disruptivity_list, TPR_list_list, FPR_list_list, predictor_list=main()
