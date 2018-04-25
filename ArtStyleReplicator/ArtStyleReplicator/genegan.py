import ctypes
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.SetConsoleTitleW("Art Style Replicator")
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
import scipy
import collections
import operator
import random
import xml.etree.ElementTree as ET
from keras.preprocessing.image import ImageDataGenerator
from win32 import win32gui

# Setting up global variables
train_data_width = 650
train_data_height = 450
z_size = 32
Max_ID_Number = -1
genimg = "../../UnityScreenshots/"
XMLPath = "../../ToonRendering/Assets/GeneticOutput.xml"
vidimg = "../../TrainingImages/"

train_datagen = ImageDataGenerator()

# Fetching training data
train_data = train_datagen.flow_from_directory(
	vidimg,
	target_size=(train_data_height, train_data_width),
	batch_size=z_size,
	class_mode=None)

# The data structure used as a base individual
Individual = collections.OrderedDict({
	'idnumber' : -1,
	'gene' : collections.OrderedDict({}),
	'image' : tf.placeholder(dtype=tf.float32, shape=[1,train_data_height, train_data_width,3]),
	'fitness' : 0.0
	})

# Leaky function to account for slight inaccuracies
def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

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

# Define genetic/generator network
def genetic(z):

	#################################################################################
	######### Functions #################

	# Ordering population by fitness
	def computePerfPopulation(population):
		result = sorted(population, key=operator.itemgetter('fitness'), reverse=True)
		return result

	# Selection for breeding
	def selectFromPopulation(populationSorted, best_sample, lucky_few):
		nextGen = []
		for i in range(best_sample):
			nextGen.append(populationSorted[i])
		for i in range(lucky_few):
			nextGen.append(random.choice(populationSorted))
		random.shuffle(nextGen)
		return nextGen

	# Breeding (individual)
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
		print("Instantiated child from XML")

		for i in range(len(child['gene'].keys())):
			if (int(100 * random.random()) < 50):
				child['gene'][str(list(child['gene'].keys())[i])] = individual1['gene'][str(list(individual1['gene'].keys())[i])]
			else:
				child['gene'][str(list(child['gene'].keys())[i])] = individual2['gene'][str(list(individual2['gene'].keys())[i])]

		generateImage(child)
		checkFitness(child)

		return child

	# Breeding (batch)
	def createChildren(breeders, number_of_child):
		print("Creadted children")
		nextPopulation = []
		for i in range(int(len(breeders)/2)):
			for j in range(number_of_child):
				nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
		return nextPopulation

	# Mutation (individual)
	def mutateGenes(individual):
		index_modification = int(random.random() * len(individual['gene'].values()))
		individual['gene'][str(list(individual['gene'].keys())[index_modification])] = random.random()
		return individual
	
	# Mutation (batch)
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

	# Evolving through an infinite number of generations until overall fitness is high enough
	def multipleGenerationFromFitness(best_sample, lucky_few, number_of_child, chance_of_mutation):
		c_fitness = 0.0
		c_fitness = c_fitness - 100000
		number_of_multigen = 0
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
			print("Overall fitness in generation " + str(i) + ": " + str(c_fitness))
		number_of_multigen = i+1
		print("Multigen from fitness passed: " + str(c_fitness))
		return historic

	# Get the best individuals sorted by fitness from the most recent population
	def getListBestIndividualFromHistorique (historic):
		return computePerfPopulation(historic[len(historic)-1])

	# Print result
	def printSimpleResult(historic, number_of_generation):
		print ("Number of generations: " + str(number_of_generation + 1))
		result = getListBestIndividualFromHistorique(historic)
		print ("Best solution has fitness: " + str(result[0]['fitness']))

	# Generator-scope variables
	best_sample = 8				# How many of the best will get chosen
	lucky_few = 8				# How many random choices also get chosen
	number_of_child = 4			# How many children does each individual have 
								# ( z_size / ((best_sample + lucky_few) / 2) ) for a stable population
	number_of_multigen = 0		# How many cycles has multigen gone through
	chance_of_mutation = 20		# How likely mutation will occur (x/100)

	#################################################################################

	# Check if population is stable
	if ((best_sample + lucky_few) / 2 * number_of_child != z_size):
		print ("Population size not stable")
	else:
		historic = multipleGenerationFromFitness(best_sample, lucky_few, number_of_child, chance_of_mutation)
		printSimpleResult(historic, number_of_multigen)

		global train_data_width
		global train_data_height

		best_list = getListBestIndividualFromHistorique(historic)

		best_out = best_list[0]['image']

		reuse_out = list()
		fitness_out = np.array(0.0)
		fitness_out = np.resize(fitness_out, [z_size, 1])

		for i in range(0, len(best_list)):
			reuse_out.append(best_list[i])
			fitness_out[i] = float(best_list[i]['fitness'])

		print("Geneticised")

	# Return:
	# The best individual, to pass to the discriminator for training
	# The [z_size] best individuals, to reuse as an input z_vector next time
	# The fitness values of the [z_size] best 
	return best_out, reuse_out, fitness_out

###################################################################################################################


###################################################################################################################
########################## Genetic helper functions ###############################################################

# Initial population generation (individual)
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

# Initial population generation (batch)
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

# Outsourcing image rendering to Unity
def generateImage(word):
	saveToXML(word)

	print("Waiting for Unity to generate image...")

	hwnd = 0
	hwnd = win32gui.FindWindow(None, "ToonRendering")
	if hwnd == 0:
		def callback(h, extra):
			if "ToonRendering" in win32gui.GetWindowText(h):
				extra.append(h)
			return True
		extra = []
		win32gui.EnumWindows(callback, extra)
		if extra: hwnd = extra[0]
	if hwnd == 0:
		print("Unity window 'ToonRendering' not found. Please ensure Unity is running and switch to it manually.")
	win32gui.SetForegroundWindow(hwnd)

	global genimg

	while True:
		if os.path.isfile(genimg + str(word['idnumber']) + ".png"):
			break
	imageString = tf.read_file(genimg + str(word['idnumber']) + ".png")

	hwnd = 0
	hwnd = win32gui.FindWindow(None, "Art Style Replicator")
	if hwnd == 0:
		def callback(h, extra):
			if "Art Style Replicator" in win32gui.GetWindowText(h):
				extra.append(h)
			return True
		extra = []
		win32gui.EnumWindows(callback, extra)
		if extra: hwnd = extra[0]
	if hwnd == 0:
		print("Python window 'Art Style Replicator' not found.")
	win32gui.SetForegroundWindow(hwnd)

	imageString = tf.image.decode_png(imageString, dtype=tf.uint16)
	imageString = tf.image.convert_image_dtype(imageString, tf.float32)

	############### Convert word['image'] to tensor4 ################
	global train_data_width
	global train_data_height
	word['image'] = tf.reshape(imageString, [1, train_data_height, train_data_width, 3])

	print("Retrieved generated image")

	return word['image']

# Checking fitness (individual)
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

# Setting up terms and links

tree = ET.parse(XMLPath)
root = tree.getroot()
for child in root:
	if(child.tag == "IDNumber"):
		child.text = str(-1)
	for subChild in child:
		subChild.text = str(0.0)

tree.write(XMLPath)

# Ensure Unity is open so the program can automate itself
input("Open and start running 'ToonRendering.exe', then press Enter to continue...")

tf.reset_default_graph()

# Initializes all the weights of the network
initializer = tf.truncated_normal_initializer(stddev=0.02)

# Placeholders for network input
z_in = tf.placeholder(shape=[z_size, None],dtype=tf.float32, name = "z_in")			# Random vector
real_in = tf.placeholder(shape=[z_size, train_data_height, train_data_width, 3],dtype=tf.float32, name = "real_in")		# Real images

Gz, _, _ = genetic(generateZVector(z_size))							# Generates images from random z vectors
Dx = discriminator(real_in, reuse=tf.AUTO_REUSE)					# Produces probabilities for real images
Dg = discriminator(Gz,reuse=True)									# Produces probabilities for generator images

# Optimization objective functions
d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.0-Dg))				# This reduces the discriminator to a single trainable value.
tvars = tf.trainable_variables()

# Applies gradient descent
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
d_grads = trainerD.compute_gradients(d_loss,tvars[0:])				# Only update the weights for the discriminator network.
update_D = trainerD.apply_gradients(d_grads)						# The generator updates itself through geneticism.

###################################################################################################################



###################################################################################################################
####################### Training the network ######################################################################

iterations = 500													# Total number of iterations to use.
sample_directory = './figs'											# Directory to save sample images from generator in.
model_directory = './models'										# Directory to save trained model to.
init = tf.global_variables_initializer()
saver = tf.train.Saver()
gz_checkpoint = generateZVector(z_size)

with tf.Session() as sess:  
	sess.run(init)
	for i in range(iterations):
		_, gz_checkpoint, fitness_list = genetic(gz_checkpoint)
		ys = train_data.next()
		xs = (np.reshape(ys, [z_size, train_data_height, train_data_width, 3]) - 0.5) * 2.0			# Transform it to be between -1 and 1
		xs = np.lib.pad(xs, ((0, 0), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=(-1, -1))	# Pad the images if necessary
		_ , dLoss = sess.run([update_D, d_loss], feed_dict={z_in:fitness_list, real_in:xs})			# Update the discriminator

###################################################################################################################