import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import slim
import sys 
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SLIM_PATH = './libs/slim-vgg16/'
IMAGENET_VAL_PATH = './raw-data/validation/'

# codebook of weights
codebook_path = './codebooks_vgg16/weights/'

# codebook of activations
CODEBOOK_PATH = './codebooks_vgg16/activations/'

bit_allocation_path = './bit_allocation_vgg16/'

sys.path.append(SLIM_PATH)

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

def load_codebook(codebook_file):
	data = read_all_lines(codebook_file)
	cluster = []
	cnt = 0
	for eachline in data[0:]:
	    c = float(eachline)
	    cluster.append(c)
	    cnt = cnt + 1

	cluster_70000 = 10e6*np.ones(shape=[70000,])

	cluster_70000[0] = cnt
	for i in range(len(cluster)):
	    cluster_70000[i + 1] = cluster[i]

	return cluster_70000

def load_bits_allocation(input_file):
	data = read_all_lines(input_file)
	bits = []
	for eachline in data[0:]:
		value = int(eachline)
		bits.append(value)
	return bits

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

def load_codebook_wei(codebook_file):
    data = read_all_lines(codebook_file)
    cluster = []
    for eachline in data[0:]:
        c = float(eachline)
        cluster.append(c)
    return cluster

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

model='vgg_16'
batch_size=100
n_images=50000
imges_path=IMAGENET_VAL_PATH
val_file='imagenet_2012_validation_synset_labels_new_index.txt'

data = read_all_lines(val_file)
ground_truth = [int(x) for x in data]
PIE_TRUTH = [x for x in ground_truth]
checkpoint_file = './models/%s.ckpt' % (model)



str_name_layers = ['vgg_16_conv1_conv1_1_weights:0', 
'vgg_16_conv1_conv1_2_weights:0',
'vgg_16_conv2_conv2_1_weights:0',
'vgg_16_conv2_conv2_2_weights:0',
'vgg_16_conv3_conv3_1_weights:0',
'vgg_16_conv3_conv3_2_weights:0',
'vgg_16_conv3_conv3_3_weights:0',
'vgg_16_conv4_conv4_1_weights:0',
'vgg_16_conv4_conv4_2_weights:0',
'vgg_16_conv4_conv4_3_weights:0',
'vgg_16_conv5_conv5_1_weights:0',
'vgg_16_conv5_conv5_2_weights:0',
'vgg_16_conv5_conv5_3_weights:0',
'vgg_16_fc6_weights:0',
'vgg_16_fc7_weights:0',
'vgg_16_fc8_weights:0'];

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
	file_results.write(str(top_1) + ' ' + str(top_5) + '\n')
	file_results.close()

	return top_1, top_5

############################# main functions ###############################

#### laod all original feature vectors from files

#id_quantized_layer_1 = int(sys.argv[1])
#id_quantized_layer_2 = int(sys.argv[2])
id_ave_size = int(sys.argv[1])

num_layers_wei = 16
num_layers_act = 15

#### build quantized vgg graph

with slim.arg_scope(vgg_arg_scope()):
	input_string = tf.placeholder(tf.string)
	input_images = tf.read_file(input_string)
	input_images = tf.image.decode_jpeg(input_images, channels=3)
	input_images = tf.cast(input_images, tf.float32)

	processed_images = vgg_preprocessing.preprocess_image(input_images, 224, 224, is_training=False)
	processed_images = tf.expand_dims(processed_images, 0)
	logits, _, acts = vgg_16_quant_act(processed_images, num_classes=1000, is_training=False)
	probabilities = tf.nn.softmax(logits)

variables_to_restore = slim.get_variables_to_restore()
variables_weights, valuables_codebooks = preprocessing_variables(variables_to_restore)

for i in range(len(variables_weights)):
	print (str(i) + ': ' + variables_weights[i].name)

for i in range(len(valuables_codebooks)):
	print (str(i) + ': ' + valuables_codebooks[i].name)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess =  tf.Session(config=config)
sess.run(tf.global_variables_initializer())
init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_weights)
init_fn(sess)

#### eval on imagenet

file_results = open('results_vgg_act_mse_out_' + str(id_ave_size) + '.txt' , 'w')

bit_allocation = load_bits_allocation(bit_allocation_path + 'bits_allocation_barget_' + str(id_ave_size) + '.txt')

for i in range(num_layers_wei):
	codebook = load_codebook_wei(codebook_path + str_name_layers[i] + '_codebook_level_' + str(bit_allocation[i]))
	weights_value = sess.run(variables_weights[i * 2])
	quantized_weights_value = map_matrix_to_codebook_uni_scalar(weights_value , codebook)
	set_variable_to_tensor(sess , variables_weights[i * 2] , quantized_weights_value)

for i in range(num_layers_act):
	CODEBOOK_DIR_2 = CODEBOOK_PATH + 'act_layer_' + str(i + 1) + '.dat' + '_codebook_level_%d' % (bit_allocation[i + 16])
	codebook_2 = load_codebook(CODEBOOK_DIR_2)
	set_variable_to_tensor(sess , valuables_codebooks[i] , codebook_2)

output_layer_id_quantized = [0] * n_images
output_layer_final_quantized = [0] * n_images
cnt = 0

############################ test ##################################
print('quantization ave size %s' % (id_ave_size))
filename = "results_" + str(id_ave_size) + ".txt"
eval_on_imagenet(filename)
