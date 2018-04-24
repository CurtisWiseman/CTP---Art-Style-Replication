import ctypes
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.SetConsoleTitleW("Art Style Replicator")

import tensorflow as tf
import numpy as np
import datetime
#from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy
import collections
import operator
import random
import xml.etree.ElementTree as ET
from keras.preprocessing.image import ImageDataGenerator
from win32 import win32gui
import re

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

class WindowMgr:
	"""Encapsulates some calls to the winapi for window management"""

	def __init__ (self):
		"""Constructor"""
		self._handle = None

	def find_window(self, class_name, window_name=None):
		"""find a window by its class_name"""
		self._handle = win32gui.FindWindow(class_name, window_name)

	def find_window_wildcard(self, wildcard):
		"""find a window whose title matches the wildcard regex"""
		self._handle = None
		win32gui.EnumWindows(self._window_enum_callback, wildcard)

	def set_foreground(self):
		"""put the window in the foreground"""
		win32gui.SetForegroundWindow(self._handle)

	def set_python_window():
		"""Gets this python window and sets it to the foreground."""
		find_window_wildcard(".*Art Style Replicator*.")
		set_foreground()

	def set_unity_window():
		"""Gets the unity window and sets it to the foreground."""
		find_window_wildcard(".*Unity*.")
		set_foreground()

	def window_callback(hwnd, cookie):
		return None

train_data_width = 650
train_data_height = 450
z_size = 32

vidimg = "../../TrainingImages/"
train_datagen = ImageDataGenerator()
train_data = train_datagen.flow_from_directory(
	vidimg,
	target_size=(train_data_height, train_data_width),
	batch_size=z_size,
	class_mode=None)
genimg = "../../UnityScreenshots/"
XMLPath = "../../ToonRendering/Assets/GeneticOutput.xml"
Max_ID_Number = -1

# The data structure used as a z-vector

Individual = collections.OrderedDict({
	'idnumber' : -1,
	'gene' : collections.OrderedDict({}),
	'image' : tf.placeholder(dtype=tf.float32, shape=[1,train_data_height, train_data_width,3]),
	'fitness' : 0.0
	})

# Leaky function to account for slight innacuracies
def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

# Save images (abstract)
def save_images(images, size, image_path):
	return imsave(inverse_transform(images), size, image_path)

# Save images (actual)
def imsave(images, size, path):
	return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
	return (images+1.)/2.

# Put images together into one structure
def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	img = np.zeros((h * size[0], w * size[1]))

	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		img[j*h:j*h+h, i*w:i*w+w] = image

	return img

# Define discriminator network
def discriminator(bottom, reuse=False):
	dis1 = slim.conv2d(bottom,
					16,
					[4,4],
					stride = [2,2],
					padding = "SAME",\
					biases_initializer = None,
					activation_fn = lrelu,\
					reuse = reuse,
					scope = 'd_conv1',
					weights_initializer = initializer)

	dis2 = slim.conv2d(dis1,
					32,
					[4,4],
					stride=[2,2],
					padding="SAME",\
					normalizer_fn=slim.batch_norm,
					activation_fn=lrelu,\
					reuse=reuse,
					scope='d_conv2',
					weights_initializer=initializer)

	dis3 = slim.conv2d(dis2,
					64,
					[4,4],
					stride=[2,2],
					padding="SAME",\
					normalizer_fn=slim.batch_norm,
					activation_fn=lrelu,\
					reuse=reuse,
					scope='d_conv3',
					weights_initializer=initializer)

	d_out = slim.fully_connected(slim.flatten(dis3),
							  1,
							  activation_fn=tf.nn.sigmoid,\
							  reuse=reuse,
							  scope='d_out',
							  weights_initializer=initializer)
	print("Discriminated")
	return d_out

#Genetirator algoritwork
def genetic(z):

	#################################################################################
	

	def computePerfPopulation(population):

		#sort_on = 'fitness'
		#decorated = [(dict_[sort_on], dict_) for dict_ in population]
		#decorated.sort()
		#result = [dict_ for (key, dict_) in decorated]

		result = sorted(population, key=operator.itemgetter('fitness'), reverse=True)
		
		return result

	# Selection
	def selectFromPopulation(populationSorted, best_sample, lucky_few):
		
		nextGen = []
		for i in range(best_sample):
			nextGen.append(populationSorted[i])
		for i in range(lucky_few):
			nextGen.append(random.choice(populationSorted))
		random.shuffle(nextGen)
		return nextGen

	# Breeding
	def createChild(individual1, individual2):
		child = dict(Individual)
		global Max_ID_Number
		Max_ID_Number += 1
		child['idnumber'] = int(Max_ID_Number)

		child['gene'] = collections.OrderedDict({})

		global XMLPath
		tree = ET.parse(XMLPath)
		root = tree.getroot()

		for node in root:
			for subNode in node:
				child['gene'][subNode.tag] = 0.0
		print("Read from XML")

		for i in range(len(child['gene'].keys())):
			if (int(100 * random.random()) < 50):
				child['gene'][str(list(child['gene'].keys())[i])] = individual1['gene'][str(list(individual1['gene'].keys())[i])]
			else:
				child['gene'][str(list(child['gene'].keys())[i])] = individual2['gene'][str(list(individual2['gene'].keys())[i])]

		generateImage(child)
		checkFitness(child)

		return child

	def createChildren(breeders, number_of_child):
		print("Creadted children")
		nextPopulation = []
		for i in range(int(len(breeders)/2)):
			for j in range(number_of_child):
				nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
		return nextPopulation

	# Mutation
	def mutateGenes(individual):
		index_modification = int(random.random() * len(individual['gene'].values()))
		individual['gene'][str(list(individual['gene'].keys())[index_modification])] = random.random()
		return individual
	
	def mutatePopulation(population, chance_of_mutation):
		for i in range(len(population)):
			if random.random() * 100 < chance_of_mutation:
				population[i] = mutateGenes(population[i])
		return population

	# Evolving a single generation
	def nextGeneration (firstGeneration, best_sample, lucky_few, number_of_child, chance_of_mutation):
		 populationSorted = computePerfPopulation(firstGeneration)
		 nextBreeders = selectFromPopulation(populationSorted, best_sample, lucky_few)
		 nextPopulation = createChildren(nextBreeders, number_of_child)
		 nextGen = mutatePopulation(nextPopulation, chance_of_mutation)
		 print("Made next generation")
		 return nextGen

	# Evolving through a set number of generations
	def multipleGeneration(number_of_generation, best_sample, lucky_few, number_of_child, chance_of_mutation):
		historic = []
		historic.append(generateZVector(z_size))
		for i in range (number_of_generation):
			historic.append(nextGeneration(historic[i], best_sample, lucky_few, number_of_child, chance_of_mutation))
		return historic

	# Evolving through an infinite number of generations until overall fitness is high enough
	def multipleGenerationFromFitness(best_sample, lucky_few, number_of_child, chance_of_mutation):
		c_fitness = 0.0
		c_fitness = c_fitness - 100000
		i = 0
		historic = []
		historic.append(z)

		while c_fitness < -10.0*z_size:
			historic.append(nextGeneration(historic[i], best_sample, lucky_few, number_of_child, chance_of_mutation))
			this_generation = historic[i]
			c_fitness = 0.0
			for j in range (z_size):
				c_fitness += float(this_generation[j]['fitness'])
			i += 1
			print("Overall fitness: " + str(c_fitness))
		number_of_multigen = i+1
		print("Multigen from fitness passed: " + str(c_fitness))
		return historic


	def getListBestIndividualFromHistorique (historic):
		return computePerfPopulation(historic[len(historic)-1])

	# Print result
	def printSimpleResult(historic, number_of_generation): #bestSolution in historic. Caution not the last
		print ("Number of generations: " + str(number_of_generation))
		result = getListBestIndividualFromHistorique(historic)
		print ("Best solution has fitness: " + str(result[0]['fitness']))

	# Variables
	best_sample = 2 # How many of the best will get chosen
	lucky_few = 2 # How many random choices also get chosen
	number_of_child = 4 # How many children does each individual have ( z_size / ((best_sample + lucky_few) / 2) ) for a stable population
	number_of_generation = 50 # How many cycles to go through
	number_of_multigen = 0 #How many cycles has multigen gone through
	chance_of_mutation = 5 # How likely mutation will occur

	#################################################################################

	#save out the genetic bit to a shader
	#CHECK - render the test scene with the shader in [unity]
	#bring the image back here as g_out

	if ((best_sample + lucky_few) / 2 * number_of_child != z_size):
		print ("Population size not stable")

	else:
		historic = multipleGenerationFromFitness(best_sample, lucky_few, number_of_child, chance_of_mutation)
		printSimpleResult(historic, number_of_multigen)
		number_of_multigen = 0

		global train_data_width
		global train_data_height

		best_list = getListBestIndividualFromHistorique(historic)

		#g_out = tf.placeholder(dtype=tf.float32, shape=[0, train_data_height, train_data_width, 3], name = "g_out")
		g_out = best_list[0]['image']

		r_out = list()
		f_out = np.array(0.0)
		f_out = np.resize(f_out, [z_size, 1])

		for i in range(0, len(best_list)):
			#g_out = tf.concat([g_out, individual['image']], 0)
			r_out.append(best_list[i])
			f_out[i] = float(best_list[i]['fitness'])

		print("Geneticised")

	return g_out, r_out, f_out

###################################################################################################################


###################################################################################################################
########################## Genetic helper functions ###############################################################

# Initial population generation
def generateZValue():
	result = dict(Individual)

	global Max_ID_Number
	Max_ID_Number += 1
	result['idnumber'] = int(Max_ID_Number)

	i = 0
	result['gene'] = collections.OrderedDict({})
	global XMLPath
	tree = ET.parse(XMLPath)
	root = tree.getroot()

	for child in root:
		for subChild in child:
			result['gene'][subChild.tag] = 0.0
	print("Read from XML")

	while i < len(result['gene']):
		part = random.random()
		keyname = str(list(result['gene'].keys())[i])
		result['gene'][keyname] = part
		i +=1

	generateImage(result)
	checkFitness(result)

	print("Generated random individual")

	return result

def generateZVector(sizePopulation):
	population = []
	i = 0
	while i < sizePopulation:
		population.append(generateZValue())
		i+=1

	print("Generated batch of individuals")

	return population

# Saving out to XML and grabbing image back
def saveToXML(word):
	global XMLPath
	tree = ET.parse(XMLPath)
	root = tree.getroot()

	for child in root:
		if(child.tag == "IDNumber"):
			child.text = str(word['idnumber'])
		for subChild in child:
			subChild.text = str(word['gene'][subChild.tag])

	tree.write(XMLPath)

	global genimg
	tree.write(genimg + str(word['idnumber']) + ".xml")

	print("Saved to XML")

def generateImage(word):
	saveToXML(word)

	print("Waiting for Unity to generate image...")

	#w = WindowMgr()
	#w.set_unity_window

	#hwnd = WindowMgr

	hwnd = 0
	window_name = "ToonRendering"

	if window_name is not None:
		hwnd = win32gui.FindWindow(None, window_name)
		if hwnd == 0:
			def callback(h, extra):
				if window_name in win32gui.GetWindowText(h):
					extra.append(h)
				return True
			extra = []
			win32gui.EnumWindows(callback, extra)
			if extra: hwnd = extra[0]
		if hwnd == 0:
			raise WindowsAppNotFoundError("Windows Application <%s> not found!" % window_name)

	#hwnd = win32gui.FindWindow(None, "Unity*.")
	win32gui.SetForegroundWindow(hwnd)

	global genimg

	while True:
		if os.path.isfile(genimg + str(word['idnumber']) + ".png"):
			break
	imageString = tf.read_file(genimg + str(word['idnumber']) + ".png")

	hwnd = win32gui.FindWindow(None, "Art Style Replicator")
	win32gui.SetForegroundWindow(hwnd)

	imageString = tf.image.decode_png(imageString, dtype=tf.uint16)
	imageString = tf.image.convert_image_dtype(imageString, tf.float32)
	############### Convert word['image'] to tensor4 ################
	global train_data_width
	global train_data_height
	word['image'] = tf.reshape(imageString, [1, train_data_height, train_data_width, 3])

	print("Retrieved generated image")
	#w.set_python_window

	return word['image']

# Checking fitness
def checkFitness(solution):
	solution['fitness'] = tf.reduce_mean(tf.log(discriminator(solution['image'], reuse=tf.AUTO_REUSE)))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		fitnessVal = solution['fitness'].eval()
		solution['fitness'] = fitnessVal

	return fitnessVal

###################################################################################################################


###################################################################################################################
######################### Connecting things together ##############################################################

tree = ET.parse(XMLPath)
root = tree.getroot()
for child in root:
	if(child.tag == "IDNumber"):
		child.text = str(-1)
	for subChild in child:
		subChild.text = str(0.0)

tree.write(XMLPath)

input("Open and start running Unity project, then press Enter to continue...")

tf.reset_default_graph()

# Initializes all the weights of the network
initializer = tf.truncated_normal_initializer(stddev=0.02)

# Placeholders for network input
z_in = tf.placeholder(shape=[z_size, None],dtype=tf.float32, name = "z_in")			# Random vector
real_in = tf.placeholder(shape=[z_size, train_data_height, train_data_width, 3],dtype=tf.float32, name = "real_in")		# Real images

Gz, _, _ = genetic(generateZVector(z_size))													# Generates images from random z vectors
Dx = discriminator(real_in, reuse=tf.AUTO_REUSE)					# Produces probabilities for real images
Dg = discriminator(Gz,reuse=True)									# Produces probabilities for generator images

# Optimization objective functions
d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.0-Dg))				# This reduces the discriminator to a single trainable value.
g_loss = -tf.reduce_mean(tf.log(Dg))								# This reduces the generator to a trainable value.

tvars = tf.trainable_variables()

# Applies gradient descent
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
trainerG = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
d_grads = trainerD.compute_gradients(d_loss,tvars[0:])				# Only update the weights for the discriminator network.

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
g_grads = trainerG.compute_gradients(g_loss,tvars[0:])	# Only update the weights for the generator network.
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)

###################################################################################################################



###################################################################################################################
####################### Training the network ######################################################################

iterations = 10000							# Total number of iterations to use.
sample_directory = './figs'					# Directory to save sample images from generator in.
model_directory = './models'				# Directory to save trained model to.
init = tf.global_variables_initializer()
saver = tf.train.Saver()
gz_checkpoint = generateZVector(z_size)

with tf.Session() as sess:  
	sess.run(init)
	for i in range(iterations):
		zs, gz_checkpoint, fitness_list = genetic(gz_checkpoint)

		ys = train_data.next()
		xs = (np.reshape(ys, [z_size, train_data_height, train_data_width, 3]) - 0.5) * 2.0			# Transform it to be between -1 and 1
		xs = np.lib.pad(xs, ((0, 0), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=(-1, -1))		# Pad the images so they are 32x32

		_ , dLoss = sess.run([update_D, d_loss], feed_dict={z_in:fitness_list, real_in:xs})						# Update the discriminator
		#_ , gLoss = sess.run([update_G, g_loss], feed_dict={z_in:zs})									# Update the generator, twice for good measure.
		#_ , gLoss = sess.run([update_G, g_loss], feed_dict={z_in:zs})


###################################################################################################################