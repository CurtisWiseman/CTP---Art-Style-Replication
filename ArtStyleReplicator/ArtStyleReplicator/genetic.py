import random
import operator
import time
import matplotlib.pyplot as plt

temps1 = time.time()


# Fitness function (replace with output from discriminator)
def fitness (password, test_word):
	score = 0
	i = 0
	while (i < len(password)):
		if (password[i] == test_word[i]):
			score+=1
		i+=1
	return score * 100 / len(password)


# Initial population generation
def generateAWord (length):
	i = 0
	result = ""
	while i < length:
		letter = chr(97 + int(26 * random.random()))
		result += letter
		i +=1
	return result

def generateFirstPopulation(sizePopulation, password):
	population = []
	i = 0
	while i < sizePopulation:
		population.append(generateAWord(len(password)))
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
	child = ""
	for i in range(len(individual1)):
		if (int(100 * random.random()) < 50):
			child += individual1[i]
		else:
			child += individual2[i]
	return child

def createChildren(breeders, number_of_child):
	nextPopulation = []
	for i in range(int(len(breeders)/2)):
		for j in range(number_of_child):
			nextPopulation.append(createChild(breeders[i], breeders[len(breeders) -1 -i]))
	return nextPopulation


# Mutation
def mutateWord(word):
	index_modification = int(random.random() * len(word))
	if (index_modification == 0):
		word = chr(97 + int(26 * random.random())) + word[1:]
	else:
		word = word[:index_modification] + chr(97 + int(26 * random.random())) + word[index_modification+1:]
	return word
	
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


# Plot a graph of overall fitness
def evolutionBestFitness(historic, password):
	plt.axis([0,len(historic),0,105])
	plt.title(password)
	
	evolutionFitness = []
	for population in historic:
		evolutionFitness.append(getBestIndividualFromPopulation(population, password)[1])
	plt.plot(evolutionFitness)
	plt.ylabel('fitness best individual')
	plt.xlabel('generation')
	plt.show()

def evolutionAverageFitness(historic, password, size_population):
	plt.axis([0,len(historic),0,105])
	plt.title(password)
	
	evolutionFitness = []
	for population in historic:
		populationPerf = computePerfPopulation(population, password)
		averageFitness = 0
		for individual in populationPerf:
			averageFitness += individual[1]
		evolutionFitness.append(averageFitness/size_population)
	plt.plot(evolutionFitness)
	plt.ylabel('Average fitness')
	plt.xlabel('generation')
	plt.show()


# Variables
password = "banana" # Used for comparison to define fitness, not needed later
size_population = 100 # How many individuals to work with
best_sample = 20 # How many of the best will get chosen
lucky_few = 20 # How many random choices also get chosen
number_of_child = 5 # How many children does each individual have ( size_population / ((best_sample + lucky_few) / 2) ) for a stable population
number_of_generation = 50 # How many cycles to go through
chance_of_mutation = 5 # How likely mutation will occur


# Runtime program, checks first for stable population variables
if ((best_sample + lucky_few) / 2 * number_of_child != size_population):
	print ("population size not stable")
else:
	historic = multipleGeneration(number_of_generation, password, size_population, best_sample, lucky_few, number_of_child, chance_of_mutation)
	
	printSimpleResult(historic, password, number_of_generation)
	
	evolutionBestFitness(historic, password)
	evolutionAverageFitness(historic, password, size_population)

print (time.time() - temps1)