import tensorflow as tf
import numpy as np
import datetime
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy


mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
vidimg = "Video_frames/"

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

# Put images together into one file
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
					stride=[2,2],
					padding="SAME",\
					biases_initializer=None,
					activation_fn=lrelu,\
					reuse=reuse,
					scope='d_conv1',
					weights_initializer=initializer)

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

	return d_out

#Define generator network
def generator(z):
	
	zP = slim.fully_connected(z,4*4*256,normalizer_fn=slim.batch_norm,\
		activation_fn=tf.nn.relu,scope='g_project',weights_initializer=initializer)
	zCon = tf.reshape(zP,[-1,4,4,256])
	
	gen1 = slim.convolution2d_transpose(\
		zCon,num_outputs=64,kernel_size=[5,5],stride=[2,2],\
		padding="SAME",normalizer_fn=slim.batch_norm,\
		activation_fn=tf.nn.relu,scope='g_conv1', weights_initializer=initializer)
	
	gen2 = slim.convolution2d_transpose(\
		gen1,num_outputs=32,kernel_size=[5,5],stride=[2,2],\
		padding="SAME",normalizer_fn=slim.batch_norm,\
		activation_fn=tf.nn.relu,scope='g_conv2', weights_initializer=initializer)
	
	gen3 = slim.convolution2d_transpose(\
		gen2,num_outputs=16,kernel_size=[5,5],stride=[2,2],\
		padding="SAME",normalizer_fn=slim.batch_norm,\
		activation_fn=tf.nn.relu,scope='g_conv3', weights_initializer=initializer)
	
	g_out = slim.convolution2d_transpose(\
		gen3,num_outputs=1,kernel_size=[32,32],padding="SAME",\
		biases_initializer=None,activation_fn=tf.nn.tanh,\
		scope='g_out', weights_initializer=initializer)
	
	return g_out

# The data structure used as a z-vector
geneStruct = {
	'ambient': 1.0, 
	'albedo': 1.0, 
	'specular': 1.0, 
	'cel': 1.0
	}

#Genetirator algoritwork
def genetic(z):

	#################################################################################
	#do the genetic bit

	# Fitness function (replace with output from discriminator)
	#def fitness (password, test_word):
	#	score = 0
	#	i = 0
	#	while (i < len(password)):
	#		if (password[i] == test_word[i]):
	#			score+=1
	#		i+=1
	#	return score * 100 / len(password)


	# Initial population generation
	def generateAWord():
		i = 0
		result = geneStruct
		while i < len(result.keys()):
			part = random.random()
			result.keys()[i] = part
			i +=1
		return result

	def generateFirstPopulation(sizePopulation):
		population = []
		i = 0
		while i < sizePopulation:
			population.append(generateAWord())
			i+=1
		return population

	def computePerfPopulation(population, password):
		populationPerf = {}
		for individual in population:
			populationPerf[individual] = fitness(password, individual)
		return sorted(populationPerf.items(), key = operator.itemgetter(1), reverse=True)


	# Selection
	def selectFromPopulation(populationSorted, best_sample, lucky_few):
		nextGeneration = []
		for i in range(best_sample):
			nextGeneration.append(populationSorted[i][0])
		for i in range(lucky_few):
			nextGeneration.append(random.choice(populationSorted)[0])
		random.shuffle(nextGeneration)
		return nextGeneration


	# Breeding
	def createChild(individual1, individual2):
		child = geneStruct
		for i in range(len(individual1.keys())):
			if (int(100 * random.random()) < 50):
				child.keys()[i] = individual1.keys()[i]
			else:
				child.keys()[i] = individual2.keys()[i]
		return child

	def createChildren(breeders, number_of_child):
		nextPopulation = []
		for i in range(int(len(breeders)/2)):
			for j in range(number_of_child):
				nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
		return nextPopulation


	# Mutation
	def mutateGenes(individual):
		index_modification = int(random.random() * len(individual.keys()))
		individual.keys()[index_modification] = random.random()
		return individual
	
	def mutatePopulation(population, chance_of_mutation):
		for i in range(len(population)):
			if random.random() * 100 < chance_of_mutation:
				population[i] = mutateWord(population[i])
		return population


	# Evolving a single generation
	def nextGeneration (firstGeneration, password, best_sample, lucky_few, number_of_child, chance_of_mutation):
		 populationSorted = computePerfPopulation(firstGeneration, password)
		 nextBreeders = selectFromPopulation(populationSorted, best_sample, lucky_few)
		 nextPopulation = createChildren(nextBreeders, number_of_child)
		 nextGeneration = mutatePopulation(nextPopulation, chance_of_mutation)
		 return nextPopulation


	# Evolving through a set number of generations
	def multipleGeneration(number_of_generation, password, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation):
		historic = []
		historic.append(generateFirstPopulation(size_population, password))
		for i in range (number_of_generation):
			historic.append(nextGeneration(historic[i], password, best_sample, lucky_few, number_of_child, chance_of_mutation))
		return historic


	# Evolving through an infinite number of generations until overall fitness is high enough
	def multipleGenerationFromFitness(password, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation):
		c_fitness = 0
		i = 0
		historic = []
		historic.append(generateFirstPopulation(size_population, password))

		while c_fitness < 95*size_population:
			historic.append(nextGeneration(historic[i], password, best_sample, lucky_few, number_of_child, chance_of_mutation))
			this_generation = historic[i]
			c_fitness = 0
			for j in range (size_population):
				c_fitness += fitness(password, this_generation[j])
			i += 1
		return historic


	# Print result
	def printSimpleResult(historic, password, number_of_generation): #bestSolution in historic. Caution not the last
		result = getListBestIndividualFromHistorique(historic, password)[number_of_generation-1]
		print ("solution: \"" + result[0] + "\" has fitness: " + str(result[1]))


	# Retrieve overall best individual(s)
	def getBestIndividualFromPopulation (population, password):
		return computePerfPopulation(population, password)[0]

	def getListBestIndividualFromHistorique (historic, password):
		bestIndividuals = []
		for population in historic:
			bestIndividuals.append(getBestIndividualFromPopulation(population, password))
		return bestIndividuals



	# Variables
	size_population = 100 # How many individuals to work with
	best_sample = 20 # How many of the best will get chosen
	lucky_few = 20 # How many random choices also get chosen
	number_of_child = 5 # How many children does each individual have ( size_population / ((best_sample + lucky_few) / 2) ) for a stable population
	number_of_generation = 50 # How many cycles to go through
	chance_of_mutation = 5 # How likely mutation will occur


	#################################################################################

	#save out the genetic bit to a shader
	#render the test scene with the shader in [unity]
	#bring the image back here as g_out

	return g_out

###################################################################################################################



###################################################################################################################
######################### Connecting things together ##############################################################

tf.reset_default_graph()
z_size = 100
number_of_shader_parameters = 9

# Initializes all the weights of the network
initializer = tf.truncated_normal_initializer(stddev=0.02)

# Placeholders for network input
z_in = tf.placeholder(shape=[None,z_size],dtype=tf.float32)			# Random vector
real_in = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32)		# Real images

Gz = generator(z_in)												# Generates images from random z vectors
Dx = discriminator(real_in)											# Produces probabilities for real images
Dg = discriminator(Gz,reuse=True)									# Produces probabilities for generator images

# Optimization objective functions
d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg))				# This optimizes the discriminator.
g_loss = -tf.reduce_mean(tf.log(Dg))								# This optimizes the generator.

tvars = tf.trainable_variables()

# Applies gradient descent
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
trainerG = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
d_grads = trainerD.compute_gradients(d_loss,tvars[0:])				# Only update the weights for the discriminator network.

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
g_grads = trainerG.compute_gradients(g_loss,tvars[0:number_of_shader_parameters])	# Only update the weights for the generator network.
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)

###################################################################################################################



###################################################################################################################
####################### Training the network ######################################################################

batch_size = 64								# Size of image batch to apply at each iteration.
iterations = 10000							# Total number of iterations to use.
sample_directory = './figs'					# Directory to save sample images from generator in.
model_directory = './models'				# Directory to save trained model to.
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:  
	sess.run(init)
	for i in range(iterations):
		zs = np.random.uniform(-1.0, 1.0, size=[batch_size, z_size]).astype(np.float32)				# Generate a random z batch
		xs , _ = mnist.train.next_batch(batch_size)													# Draw a sample batch from MNIST dataset.
		xs = (np.reshape(xs, [batch_size, 28, 28, 1]) - 0.5) * 2.0									# Transform it to be between -1 and 1
		xs = np.lib.pad(xs, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=(-1, -1)) # Pad the images so they are 32x32

		_ , dLoss = sess.run([update_D, d_loss], feed_dict={z_in:zs, real_in:xs})					# Update the discriminator
		_ , gLoss = sess.run([update_G, g_loss], feed_dict={z_in:zs})								# Update the generator, twice for good measure.
		_ , gLoss = sess.run([update_G, g_loss], feed_dict={z_in:zs})

		if i % 10 == 0:
			print ("Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss))
			z2 = np.random.uniform(-1.0, 1.0, size=[batch_size, z_size]).astype(np.float32)			# Generate another z batch
			newZ = sess.run(Gz, feed_dict={z_in:z2})												# Use new z to get sample images from generator.

			if not os.path.exists(sample_directory):												# If directory doesn't exist, make it
				os.makedirs(sample_directory)

			save_images(																			# Save sample generator images for viewing training progress.
				np.reshape(newZ[0:36], [36, 32, 32]),
				[6, 6],
				sample_directory + '/fig' + str(i) + '.png'
				)

		if i % 1000 == 0 and i != 0:																# Save checkpoints of the model
			
			if not os.path.exists(model_directory):												# If directory doesn't exist, make it
				os.makedirs(model_directory)

			saver.save(sess, model_directory + '/model-' + str(i) + '.cptk')
			print("Saved Model")


###################################################################################################################



####################################################################################################################
######################## Using a pre-trained network ###############################################################

#sample_directory = './figs'										# Directory to save sample images from generator in.
#model_directory = './models'									# Directory to load trained model from.
#batch_size_sample = 36

#init = tf.global_variables_initializer()
#saver = tf.train.Saver()
#with tf.Session() as sess:  
#    sess.run(init)
#    # Reload the model
#    print ('Loading Model...')
#    ckpt = tf.train.get_checkpoint_state(model_directory)
#    saver.restore(sess,ckpt.model_checkpoint_path)
	
#    zs = np.random.uniform(-1.0,1.0,size=[batch_size_sample,z_size]).astype(np.float32)		# Generate a random z batch
#    #newZ = sess.run(Gz,feed_dict={z_in:z2})													# Use new z to get sample images from generator.
#    if not os.path.exists(sample_directory):
#        os.makedirs(sample_directory)
#    #save_images(np.reshape(newZ[0:batch_size_sample],[36,32,32]),[6,6],sample_directory+'/fig'+str(i)+'.png')

####################################################################################################################