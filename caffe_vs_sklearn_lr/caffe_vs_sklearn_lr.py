'''



Requirements:
sudo pip install pydot
sudo apt-get install -y graphviz

Interesting resources on Caffe:
 - https://github.com/BVLC/caffe/tree/master/examples
 - http://nbviewer.ipython.org/github/joyofdata/joyofdata-articles/blob/master/deeplearning-with-caffe/Neural-Networks-with-Caffe-on-the-GPU.ipynb
 
Interesting resources on Iris with ANNs:
 - iris data set test bed: http://deeplearning4j.org/iris-flower-dataset-tutorial.html
 - http://se.mathworks.com/help/nnet/examples/iris-clustering.html
 - http://lab.fs.uni-lj.si/lasin/wp/IMIT_files/neural/doc/seminar8.pdf
 
Synonyms:
 - output = label = target
 - input = feature 
'''

import subprocess
import platform
import copy

from sklearn.datasets import load_iris
import sklearn.metrics 
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import h5py
if platform.system() == 'Linux':
    import caffe
    import caffe.draw
    import google.protobuf 
from sklearn import linear_model


def load_data():
    '''
    Load Iris Data set
    '''
    data = load_iris()
    #print(data.data)
    #print(data.target)
    
    idx = range(50, 150)
    #idx = range(0, 100)
    #idx = range(0, 50) + range(100, 150)
    data.data = data.data[idx]# [0:100]
    data.target = data.target[idx] #[0:100]
    np.place(data.target, data.target>1, [0])
    print('data.target: {0}'.format(data.target))
    #1/0
    
    targets = np.zeros((len(data.target), 3))
    for count, target in enumerate(data.target):
        targets[count][target]= 1    
    print(targets)
    
    new_data = {}
    #new_data['input'] = data.data
    new_data['input'] = np.reshape(data.data, (data.data.shape[0], 1,1,data.data.shape[1]))
    #new_data['input'] = np.reshape(data.data[0:80, :], (80,1,1,4))
    #new_data['output'] = targets
    new_data['output'] = data.target#[0:80]
    print(targets)
    new_data['input_raw'] = data.data#[0:80, :]
    new_data['output_raw'] = data.target#[0:80]
    #print(new_data['input'].shape)
    #new_data['input'] = np.random.random((150, 1, 1, 4))
    #print(new_data['input'].shape)   
    #new_data['output'] = np.random.random_integers(0, 1, size=(150,3))    
    #print(new_data['input'])
    
    return new_data

def save_data_as_hdf5(hdf5_data_filename, data):
    '''
    HDF5 is one of the data formats Caffe accepts
    '''
    with h5py.File(hdf5_data_filename, 'w') as f:
        f['data'] = data['input'].astype(np.float32)
        f['label'] = data['output'].astype(np.float32)
    

def train(solver_prototxt_filename):
    '''
    Train the ANN
    '''
    caffe.set_mode_cpu()
    solver = caffe.get_solver(solver_prototxt_filename)
    solver.solve()
    
    
def print_network_parameters(net):
    '''
    Print the parameters of the network
    '''
    print(net)
    print('net.inputs: {0}'.format(net.inputs))
    print('net.outputs: {0}'.format(net.outputs))
    print('net.blobs: {0}'.format(net.blobs))
    print('net.params: {0}'.format(net.params))    

def get_predicted_output(deploy_prototxt_filename, caffemodel_filename, input, net = None):
    '''
    Get the predicted output, i.e. perform a forward pass
    '''
    if net is None:
        net = caffe.Net(deploy_prototxt_filename,caffemodel_filename, caffe.TEST)
        
    #input = np.array([[ 5.1,  3.5,  1.4,  0.2]])
    #input = np.random.random((1, 1, 1))
    #print(input)
    #print(input.shape)
    out = net.forward(data=input)
    #print('out: {0}'.format(out))
    return out[net.outputs[0]]



def print_network(prototxt_filename, caffemodel_filename):
    '''
    Draw the ANN architecture
    '''
    _net = caffe.proto.caffe_pb2.NetParameter()
    f = open(prototxt_filename)
    google.protobuf.text_format.Merge(f.read(), _net)
    caffe.draw.draw_net_to_file(_net, prototxt_filename + '.png' )
    print('Draw ANN done!')


def print_network_weights(prototxt_filename, caffemodel_filename):
    '''
    For each ANN layer, print weight heatmap and weight histogram 
    '''
    net = caffe.Net(prototxt_filename,caffemodel_filename, caffe.TEST)
    for layer_name in net.params: 
        # weights heatmap 
        arr = net.params[layer_name][0].data
        plt.clf()
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(arr, interpolation='none')
        fig.colorbar(cax, orientation="horizontal")
        plt.savefig('{0}_weights_{1}.png'.format(caffemodel_filename, layer_name), dpi=100, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
        plt.close()
        
        # weights histogram  
        plt.clf()
        plt.hist(arr.tolist(), bins=20)
        plt.savefig('{0}_weights_hist_{1}.png'.format(caffemodel_filename, layer_name), dpi=100, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
        plt.close()
    
    
def get_predicted_outputs(deploy_prototxt_filename, caffemodel_filename, inputs):
    '''
    Get several predicted outputs
    '''
    outputs = []
    net = caffe.Net(deploy_prototxt_filename,caffemodel_filename, caffe.TEST)
    for input in inputs:
        #print(input)
        outputs.append(copy.deepcopy(get_predicted_output(deploy_prototxt_filename, caffemodel_filename, input, net)))
    return outputs    

    
def get_accuracy(true_outputs, predicted_outputs):
    '''
    
    '''
    number_of_samples = true_outputs.shape[0]
    threshold = 0.0 # 0 if SigmoidCrossEntropyLoss ; 0.5 if EuclideanLoss
    predicted_output_binary = []
    for sample_number in range(number_of_samples):         
        #1/0           
        if predicted_outputs[sample_number][0][0] > predicted_outputs[sample_number][0][1]:
            predicted_output = 0
        else:
            predicted_output = 1
        predicted_output_binary.append(predicted_output)
        
    print('predicted_outputs: {0}'.format(predicted_outputs))
    print('accuracy: {0}'.format(sklearn.metrics.accuracy_score(true_outputs, predicted_output_binary)))
    print(sklearn.metrics.confusion_matrix(true_outputs, predicted_output_binary))
    
    
def main():
    '''
    This is the main function
    '''
    
    # Set parameters
    solver_prototxt_filename = 'iris_solver.prototxt'
    train_test_prototxt_filename = 'iris_train_test.prototxt'
    deploy_prototxt_filename  = 'iris_deploy.prototxt'
    deploy_prototxt_filename  = 'iris_deploy.prototxt'
    hdf5_train_data_filename = 'iris_train_data.hdf5' 
    hdf5_test_data_filename = 'iris_test_data.hdf5' 
    caffemodel_filename = 'iris__iter_100000.caffemodel' # generated by train()
    
    # Prepare data
    data = load_data()
    print(data)
    train_data = data
    test_data = data
    save_data_as_hdf5(hdf5_train_data_filename, data)
    save_data_as_hdf5(hdf5_test_data_filename, data)
    
    # sklearn 
    # http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
    logreg = linear_model.LogisticRegression(C=1e5)
    #logreg.fit( np.transpose(data['input']), data['output'])
    logreg.fit(data['input_raw'], data['output_raw'])
    
    
    y_pred = []
    for input in data['input_raw']:
        #print(input)
        #print(logreg.predict(input))
        y_pred.append(logreg.predict(input))
    
    
    
    
    # Train network
    train(solver_prototxt_filename)
        
    # Get predicted outputs
    input = np.array([[ 5.1,  3.5,  1.4,  0.2]])
    print(get_predicted_output(deploy_prototxt_filename, caffemodel_filename, input))
    input = np.array([[[[ 5.1,  3.5,  1.4,  0.2]]],[[[ 5.9,  3. ,  5.1,  1.8]]]])
    #print(get_predicted_output(deploy_prototxt_batch2_filename, caffemodel_filename, input))
    
    # Print network
    print_network(deploy_prototxt_filename, caffemodel_filename)
    print_network(train_test_prototxt_filename, caffemodel_filename)
    print_network_weights(train_test_prototxt_filename, caffemodel_filename)
    
    # Compute performance metrics
    #inputs = input = np.array([[[[ 5.1,  3.5,  1.4,  0.2]]],[[[ 5.9,  3. ,  5.1,  1.8]]]])
    inputs = data['input']
    outputs = get_predicted_outputs(deploy_prototxt_filename, caffemodel_filename, inputs)
    get_accuracy(data['output'], outputs)
    print("sklearn Accuracy: {0}".format(sklearn.metrics.accuracy_score(data['output_raw'], y_pred)))
    
if __name__ == "__main__":
    main()
    #cProfile.run('main()') # if you want to do some profiling