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

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

train_data_width = 650
train_data_height = 450
train_data_batch_size = 8
vidimg = "../../TrainingImages/"
train_datagen = ImageDataGenerator()
train_data = train_datagen.flow_from_directory(
	vidimg,
	target_size=(train_data_width, train_data_height),
	batch_size=train_data_batch_size,
	class_mode=None)
genimg = "../../UnityScreenshots/"
XMLPath = "../../ToonRendering/Assets/GeneticOutput.xml"
Max_ID_Number = -1

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

# The data structure used as a z-vector

geneStruct = collections.OrderedDict({
	#'_Outline': 1.0,
	#'_O_Width' : 1.0,
	#'_O_ColourRed' : 0.0,
	#'_O_ColourGreen' : 0.0,
	#'_O_ColourBlue' : 0.0,

	#'_Cel' : 1.0,
	#'_C_Levels' : 2.0,

	#'_Hatching' : 1.0,
	#'_H_Intensity' : 1.0,
	#'_H_ColourRed' : 0.0,
	#'_H_ColourGreen' : 0.0,
	#'_H_ColourBlue' : 0.0,
	})

Individual = collections.OrderedDict({
	'idnumber' : -1,
	'gene' : dict(geneStruct),
	'image' : tf.placeholder(dtype=tf.float32, shape=[1,650,450,3]),
	'fitness' : 0.0
	})

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
		
		nextGeneration = []
		for i in range(best_sample):
			nextGeneration.append(populationSorted[i])
		for i in range(lucky_few):
			nextGeneration.append(random.choice(populationSorted))
		random.shuffle(nextGeneration)
		return nextGeneration

	# Breeding
	def createChild(individual1, individual2):
		child = dict(Individual)
		global Max_ID_Number
		Max_ID_Number += 1
		child['idnumber'] = Max_ID_Number

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
		 nextGeneration = mutatePopulation(nextPopulation, chance_of_mutation)
		 print("Made next generation")
		 return nextPopulation

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
		i = 0
		historic = []
		historic.append(generateZVector(z_size))

		while c_fitness < -10.0*z_size:
			historic.append(nextGeneration(historic[i], best_sample, lucky_few, number_of_child, chance_of_mutation))
			this_generation = historic[i]
			c_fitness = 0.0
			for j in range (z_size):
				c_fitness += checkFitness(this_generation[j])
			i += 1
			print("Overall fitness: " + c_fitness)
		print("Multigen from fitness passed: " + c_fitness)
		return historic

	# Retrieve overall best individual(s)
	def getBestIndividualFromPopulation (population):
		return computePerfPopulation(population)[0]

	def getListBestIndividualFromHistorique (historic):
		bestIndividuals = []
		for population in historic:
			bestIndividuals.append(getBestIndividualFromPopulation(population))
		return bestIndividuals

	# Print result
	def printSimpleResult(historic, number_of_generation): #bestSolution in historic. Caution not the last
		result = getListBestIndividualFromHistorique(historic)[number_of_generation-1]
		print ("Best solution has fitness: " + str(result[0]))

	# Variables
	best_sample = 4 # How many of the best will get chosen
	lucky_few = 4 # How many random choices also get chosen
	number_of_child = 2 # How many children does each individual have ( z_size / ((best_sample + lucky_few) / 2) ) for a stable population
	number_of_generation = 50 # How many cycles to go through
	chance_of_mutation = 5 # How likely mutation will occur

	#################################################################################

	#save out the genetic bit to a shader
	#CHECK - render the test scene with the shader in [unity]
	#bring the image back here as g_out

	if ((best_sample + lucky_few) / 2 * number_of_child != z_size):
		print ("Population size not stable")
		g_out=0
	else:
		historic = multipleGenerationFromFitness(best_sample, lucky_few, number_of_child, chance_of_mutation)
		printSimpleResult(historic, number_of_generation)
		g_out = getListBestIndividualFromHistorique(historic)[0]

	print("Geneticised")

	return g_out

###################################################################################################################



###################################################################################################################
########################## Genetic helper functions ###############################################################

# Initial population generation
def generateZValue():
	result = dict(Individual)

	global Max_ID_Number
	Max_ID_Number += 1
	result['idnumber'] = Max_ID_Number

	i = 0
	result['gene'] = dict(geneStruct)
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

	global genimg

	while True:
		if os.path.isfile(genimg + str(word['idnumber']) + ".png"):
			break
	imageString = tf.read_file(genimg + str(word['idnumber']) + ".png")
	imageString = tf.image.decode_png(imageString, dtype=tf.uint16)
	imageString = tf.image.convert_image_dtype(imageString, tf.float32)
	############### Convert word['image'] to tensor4 ################
	word['image'] = tf.reshape(imageString, [1, 650, 450, 3])

	print("Retrieved generated image")

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


#with tf.Session() as sess:
#		sess.run(tf.global_variables_initializer())
#		reduced = tf.reduce_mean(solution['fitness'])
#		evaluated = reduced.eval()
#		fitnessVal = evaluated[0, 1]
###################################################################################################################
######################### Connecting things together ##############################################################

tf.reset_default_graph()
z_size = 8
number_of_shader_parameters = 9

# Initializes all the weights of the network
initializer = tf.truncated_normal_initializer(stddev=0.02)

# Placeholders for network input
z_in = tf.placeholder(shape=[None,z_size],dtype=tf.float32)			# Random vector
real_in = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32)		# Real images

Gz = genetic(z_in)													# Generates images from random z vectors
Dx = discriminator(real_in)											# Produces probabilities for real images
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

batch_size = 8								# Size of image batch to apply at each iteration.
iterations = 10000							# Total number of iterations to use.
sample_directory = './figs'					# Directory to save sample images from generator in.
model_directory = './models'				# Directory to save trained model to.
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:  
	sess.run(init)
	for i in range(iterations):
		#zs = np.random.uniform(-1.0, 1.0, size=[batch_size, z_size]).astype(np.float32)				# Generate a random z batch
		zs = generateZVector(z_size)
		xs , _ = train_data.next																		# Draw a sample batch from dataset.
		xs = (np.reshape(xs, [batch_size, 28, 28, 1]) - 0.5) * 2.0										# Transform it to be between -1 and 1
		xs = np.lib.pad(xs, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=(-1, -1))		# Pad the images so they are 32x32

		_ , dLoss = sess.run([update_D, d_loss], feed_dict={z_in:zs, real_in:xs})						# Update the discriminator
		_ , gLoss = sess.run([update_G, g_loss], feed_dict={z_in:zs})									# Update the generator, twice for good measure.
		_ , gLoss = sess.run([update_G, g_loss], feed_dict={z_in:zs})

		if i % 10 == 0:
			print ("Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss))
			#z2 = np.random.uniform(-1.0, 1.0, size=[batch_size, z_size]).astype(np.float32)			# Generate another z batch
			z2 = generateZVector(z_size)
			newZ = sess.run(Gz, feed_dict={z_in:z2})													# Use new z to get sample images from generator.

			if not os.path.exists(sample_directory):													# If directory doesn't exist, make it
				os.makedirs(sample_directory)

			save_images(																				# Save sample generator images for viewing training progress.
				np.reshape(newZ[0:36], [36, 32, 32]),
				[6, 6],
				sample_directory + '/fig' + str(i) + '.png'
				)

		if i % 1000 == 0 and i != 0:																	# Save checkpoints of the model
			
			if not os.path.exists(model_directory):														# If directory doesn't exist, make it
				os.makedirs(model_directory)

			saver.save(sess, model_directory + '/model-' + str(i) + '.cptk')
			print("Saved Model")


###################################################################################################################