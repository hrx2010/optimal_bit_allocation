import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import slim
import sys 
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

inf_max = 10000000

SLIM_PATH = './libs/slim-resnet50/'
IMAGENET_VAL_PATH = './data/validation/'
WEI_CODEBOOK_PATH = './codebooks_resnet50/weights/'
ACT_CODEBOOK_PATH = './codebooks_resnet50/activations/'
BIT_ALLOCATION_PATH = './bit_allocation_resnet50/'

str_name_layers = ["resnet_v1_50_conv1_weights:0",
"resnet_v1_50_block1_unit_1_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block1_unit_1_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block1_unit_1_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block1_unit_2_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block1_unit_2_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block1_unit_2_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block1_unit_3_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block1_unit_3_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block1_unit_3_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block2_unit_1_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block2_unit_1_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block2_unit_1_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block2_unit_2_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block2_unit_2_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block2_unit_2_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block2_unit_3_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block2_unit_3_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block2_unit_3_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block2_unit_4_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block2_unit_4_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block2_unit_4_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block3_unit_1_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block3_unit_1_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block3_unit_1_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block3_unit_2_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block3_unit_2_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block3_unit_2_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block3_unit_3_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block3_unit_3_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block3_unit_3_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block3_unit_4_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block3_unit_4_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block3_unit_4_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block3_unit_5_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block3_unit_5_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block3_unit_5_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block3_unit_6_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block3_unit_6_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block3_unit_6_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block4_unit_1_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block4_unit_1_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block4_unit_1_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block4_unit_2_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block4_unit_2_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block4_unit_2_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_block4_unit_3_bottleneck_v1_conv1_weights:0",
"resnet_v1_50_block4_unit_3_bottleneck_v1_conv2_weights:0",
"resnet_v1_50_block4_unit_3_bottleneck_v1_conv3_weights:0",
"resnet_v1_50_logits_weights:0"
]

sys.path.append(SLIM_PATH)
number_layers = 50

number_layers_activations = 49

from nets.resnet_v1 import *
from nets.resnet_v2 import * 
from nets.resnet_v2_combine_bn import * 
from nets.vgg import *
from utils import *
from preprocessing import inception_preprocessing 
from preprocessing import vgg_preprocessing 

def write_data_to_file(file_name, data):
	fid = open(file_name, "w")
	fid.write(data)
	fid.close()

def map_matrix_to_codebook_faster(matrix, codebook):
    org_shape = np.shape(matrix)
    codebook = np.asarray(codebook)
    codebook = np.squeeze(codebook)
    temp_mat = np.reshape(matrix, [-1])
    temp_mat = np.squeeze(temp_mat)
    len_codebook = np.shape(codebook)[0]
    len_mat = np.shape(temp_mat)[0]
    temp_mat = np.expand_dims(temp_mat, axis=1)
    codebook = np.expand_dims(codebook, axis=0)
    m = np.repeat(temp_mat, len_codebook, axis=1)
    c = np.repeat(codebook, len_mat, axis=0)
    
    assert(np.shape(m) == np.shape(c))
    d = np.abs(m - c)
    select_id = np.argmin(d, axis=1)
    
    new_mat = [c[enum, item] for enum, item in enumerate(select_id)]
    return np.reshape(new_mat, org_shape)

def map_matrix_to_codebook(matrix, codebook):
    org_shape = np.shape(matrix)
    codebook = np.asarray(codebook)
    temp_mat = np.reshape(matrix, [-1])
    new_mat = np.zeros_like(temp_mat)
    for i in range(len(temp_mat)):
        curr = temp_mat[i]
        idx = np.argmin(np.abs(codebook - curr))
        new_mat[i] = codebook[idx]
    return np.reshape(new_mat, org_shape)

def set_variable_to_tensor(sess, tensor, value):
    return sess.run(tf.assign(tensor, value))

def load_codebook_act(codebook_file , removed):
	if (removed == 1):
		cluster_1024 = 10e6*np.ones(shape=[70000,])
		cluster_1024[0] = 10000000000000.0
		cluster_1024[1] = 0.0
		cluster_1024[2] = (2**-16)
		cluster_1024[3] = -10000000000000.0

		return cluster_1024
		
	else :
		data = read_all_lines(codebook_file)
		cluster = []
		cnt = 0
		for eachline in data[0:]:
		    c = float(eachline)
		    cluster.append(c)
		    cnt = cnt + 1

		cluster_1024 = 10e6*np.ones(shape=[70000,])

		cluster_1024[0] = cnt
		cluster_1024[1] = cluster[0]
		cluster_1024[2] = cluster[1]
		cluster_1024[3] = 0

		return cluster_1024

def load_codebook_wei(codebook_file):
    data = read_all_lines(codebook_file)
    cluster = []
    for eachline in data[0:]:
        c = float(eachline)
        cluster.append(c)

    return cluster

def load_random_testing_images(list_file):
    data = read_all_lines(list_file)
    testing_images = []
    for eachline in data[0:]:
        c = int(eachline)
        testing_images.append(c)

    return testing_images

def preprocessing_variables(all_variables):
	variable_weights = []
	variable_codebooks = []
	
	for i in range(len(all_variables)):
		if "codebook" in all_variables[i].name:
			variable_codebooks.append(all_variables[i])

	for i in range(len(all_variables)):
		if "codebook" not in all_variables[i].name:
			variable_weights.append(all_variables[i])

	return variable_weights, variable_codebooks

def convert_to_1D_array(values):
	values_1D = np.reshape(values, [-1])
	return values_1D

def get_all_weights_variables(all_variables):
	variable_weights = []
	
	for i in range(len(all_variables)):
		if "weights" in all_variables[i].name and "shortcut" not in all_variables[i].name:
			variable_weights.append(all_variables[i])

	return variable_weights

def map_matrix_to_codebook_uni_scalar(matrix, codebook):
	st = codebook[0]
	base = codebook[1] - codebook[0]
	quant_level = len(codebook)

	x = np.subtract(matrix , st)
	y = np.divide(x , base)
	z = np.around(y)
	
	a = np.clip(z , 0 , quant_level - 1)
	b = np.multiply(a , base)
	
	results = np.add(b , st)

	return results

def load_bits_allocation(input_file):
	data = read_all_lines(input_file)
	bits = []
	for eachline in data[0:]:
		value = int(eachline)
		bits.append(value)
	return bits

def get_all_weights_variables(all_variables):
	variable_weights = []
	
	for i in range(len(all_variables)):
		if "weights" in all_variables[i].name and "shortcut" not in all_variables[i].name:
			variable_weights.append(all_variables[i])

	return variable_weights

model='resnet_v1_50'
batch_size=100
n_images=50000
imges_path=IMAGENET_VAL_PATH
val_file='imagenet_2012_validation_synset_labels_new_index.txt'

data = read_all_lines(val_file)
ground_truth = [int(x) for x in data]
PIE_TRUTH = [x for x in ground_truth]
checkpoint_file = './models/%s.ckpt' % (model)

ave_bit_width = sys.argv[1]


def eval_on_imagenet(filename):
	top_1 = 0 
	top_5 = 0 
	for b in range(int(np.ceil(n_images/np.float(batch_size)))):
		start = b * batch_size + 1
		stop = np.minimum(n_images + 1, start+batch_size)
		for i in range(start, stop, 1): 
			img_path = imges_path + 'ILSVRC2012_val_%08d.JPEG' % (i)

			pred = sess.run(probabilities, feed_dict={input_string:img_path})

			if i == start:
				preds = pred 
			else: 
				preds = np.concatenate([preds, pred], axis=0)

			label = np.argsort(pred[0][:])[::-1]

			if label[0] == PIE_TRUTH[i-1]:
				top_1 += 1
			for q in range(5):
				if label[q] == PIE_TRUTH[i-1]:
					top_5 += 1

		print('Process %d images on %d query, suceess %d images, %0.2f ' %\
		              (stop, n_images, top_1, 100*top_1/np.float(stop-1)))

	print('Accuraty (top 1 %d top 5 %d) top_1 %0.4f. top_5 %0.4f ' % (top_1 , top_5 , 1.0 * top_1 / n_images , 1.0 * top_5 / n_images))
	
	file_results = open(filename , 'w')
	file_results.write(str(1.0 * top_1/ n_images) + ' ' + str(1.0 * top_5/ n_images) + '\n')
	file_results.close()

	return top_1, top_5

############################# main functions ###############################
#### laod all original feature vectors from files
#### build quantized vgg graph

with slim.arg_scope(resnet_arg_scope()):
	input_string = tf.placeholder(tf.string)
	input_images = tf.read_file(input_string)
	input_images = tf.image.decode_jpeg(input_images, channels=3)
	input_images = tf.cast(input_images, tf.float32)

	processed_images = vgg_preprocessing.preprocess_image(input_images, 224, 224, is_training=False)
	processed_images = tf.expand_dims(processed_images, 0)
	logits, _ = resnet_v1_50(processed_images, num_classes=1000, is_training=False)
	probabilities = tf.nn.softmax(logits)

variables_to_restore = slim.get_variables_to_restore()
variables_weights, valuables_codebooks = preprocessing_variables(variables_to_restore)
all_cnn_weights = get_all_weights_variables(variables_to_restore)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess =  tf.Session(config=config)
sess.run(tf.global_variables_initializer())
init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_weights)
init_fn(sess)

######################### load bit allocation ##############################
bit_allocation = load_bits_allocation(BIT_ALLOCATION_PATH + 'bits_allocation_barget_' + ave_bit_width + '.txt')
#bit_allocation_act = load_bits_allocation(BIT_ALLOCATION_ACT_PATH + 'bits_allocation_barget_' + ave_bit_width_act + '.txt')

print('layers of weights %d, layers of activations %d.' % (len(all_cnn_weights) , len(valuables_codebooks)))

######################### quantization #############################
for i in range(len(all_cnn_weights)):
	codebook_wei = load_codebook_wei(WEI_CODEBOOK_PATH + all_cnn_weights[i].name.replace("/" , "_") + '_codebook_level_' + str(bit_allocation[i]))
	weights_value = sess.run(all_cnn_weights[i])
	quantized_weights_value = map_matrix_to_codebook_uni_scalar(weights_value , codebook_wei)
	set_variable_to_tensor(sess , all_cnn_weights[i] , quantized_weights_value)

for i in range(number_layers_activations):
	if bit_allocation[50 + i] != inf_max:
		codebook_act = load_codebook_act(ACT_CODEBOOK_PATH + 'act_layer_' + str(i) + '.dat' + '_codebook_level_' + str(bit_allocation[50 + i]) , 0)
		set_variable_to_tensor(sess , valuables_codebooks[i] , codebook_act)
	else :
		codebook_act = load_codebook_act(ACT_CODEBOOK_PATH + 'act_layer_' + str(i) + '.dat' + '_codebook_level_' + str(1001) , 1)
		set_variable_to_tensor(sess , valuables_codebooks[i] , codebook_act)

############################ test ##################################
print('quantization ave size %s' % (ave_bit_width))
filename = "results_" + ave_bit_width + ".txt"
eval_on_imagenet(filename)


