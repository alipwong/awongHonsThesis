from operator import attrgetter
from random import shuffle
import numpy as np
import sys
import subprocess
import itertools
import os

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

class Individual:
	'''this is an individual of the simulation. 
	Need to provide a file of settings'''
	def __init__(self, paraVals):
		self.paraVals = paraVals
		if self.stray == "para to boundary":
			self.paraToBoundary()
		if self.stray == "para randomise":
			self.paraRandomise()
		if self.stray == "ind randomise":
			self.indRandomise()
		else:
			# do nothing - let them stray out of bounds
			pass
		self.evaluate()

		self.stats = self.paraVals + [self.fitScore, self.time]
		Individual.allInd.append(self.stats)

	def paraToBoundary(self):
		for para in range(self.paraN):
			if self.paraVals[para] < self.lBound[para]:
				self.paraVals[para] = self.lBound[para]
			if self.paraVals[para] > self.uBound[para]:
				self.paraVals[para] = self.uBound[para]

	def paraRandomise(self):
		for para in range(self.paraN):
			if self.paraVals[para] < self.lBound[para] or self.paraVals[para] > self.uBound[para]:
				self.paraVals[para] = np.random.uniform(self.lBound[para], self.uBound[para])

	def indRandomise(self):
		for para in range(self.paraN):
			if self.paraVals[para] < self.lBound[para] or self.paraVals[para] > self.uBound[para]:
				self.paraVals = [np.random.uniform(self.lBound[p], self.uBound[p]) for p in range(self.paraN)]
				break


class GA:
	def __init__(self, settings):

		###
		# read in settings
		###

		f = open(settings)
		for line in f:
			line = line.strip().split(',')
				
			if line[0] == "parameter list":
				self.paraName = [para.strip() for para in line[1:]]
				self.paraN = len(self.paraName)
				Individual.paraName = self.paraName
				Individual.paraN = self.paraN
			if line[0] == "problem type":
				self.problemType = line[1].strip()

			if line[0] == "upper bounds":
				self.uBound = [float(ub.strip()) for ub in line[1:]]
				Individual.uBound = self.uBound
			if line[0] == "lower bounds":
				self.lBound = [float(lb.strip()) for lb in line[1:]]
				Individual.lBound = self.lBound

			if line[0] == "stray":
				Individual.stray = line[1].strip()

			if line[0] == "population size":
				self.popSize = int(line[1].strip())

			if line[0] == "cull rate":
				self.cullR = float(line[1].strip())

			if line[0] == "crossover rate":
				self.crossoverR = float(line[1].strip())
			if line[0] == "crossover method":
				if line[1].strip() == "multi-point":
					self.crossoverMethod = "multi-point"
					self.crossover = self.multiPoint
					self.broodN = int(line[2].strip())
					self.surviveN = int(line[3].strip())

			if line[0] == "selection method":
				if line[1].strip() == "tournament":
					self.selectionMethod = "tournament"
					self.select = self.tournament
					self.winRate = float(line[2])
					self.participants = int(line[3])
				if line[1].strip() == "roulette":
					self.selectionMethod = "rouletteWheel"
					self.select = self.rouletteWheel
					if len(line) == 3:
						self.markerDist = line[2].strip()
					else:
						self.markerDist = None

			if line[0] == "clone rate":
				self.cloneR = float(line[1].strip())

			if line[0] == "mutation rate":
				self.mutateR = float(line[1].strip())
			if line[0] == "mutation method":
				if line[1].strip() == "random":
					self.mutationMethod = "random"
					self.mutate = self.randomMutation
			if line[0] == "mutation selection":
				self.mutationSelect = line[1].strip()

			if line[0] == "stopping criteria":
				self.stoppingCriteria = line[1].strip()
				if self.stoppingCriteria == "epochs":
					self.epochs = int(line[2].strip())
				if self.stoppingCriteria == "best":
					self.epsilon = float(line[2].strip())
					self.epochs = int(line[3].strip())

			if line[0] == "results folder":
				self.resFolder = line[1].strip()
				Individual.resFolder = self.resFolder
				# Check to see if results folder already exists or not
				if os.path.exists(self.resFolder) == False:
					print("Making results folder")
					subprocess.call(["mkdir", self.resFolder])
				else:
					print(self.resFolder + " already exists and contains:")
					subprocess.call(["ls", self.resFolder])
					print("Are you sure you wish to continue? (Y/N)")
					userInput = input("(Warning: all contents will be lost)\n")
					if userInput == 'Y':
						continue
					else:
						sys.exit()
			if line[0] == "save data":
				if line[1].strip() == 'T':
					self.saveData = True

		f.close()

		###
		# Some basic checks
		###

		if self.problemType != "minimise":
			if self.problemType != "maximise":
				print("Error: invalid settings file.\nSpecify whether the problem type is to minimise or maximise the fitness score.")
				sys.exit()

		if len(self.paraName) != len(self.uBound) or len(self.paraName) != len(self.lBound):
			print("Error: invalid settings file.\nThe number of parameter bounds provided does not match the number of parameters.")
			sys.exit()

		if self.broodN < self.surviveN:
			print("Error: invalid settings file.\nMust produce at least as many offspring as the number that survive.")
			sys.exit()

		if self.selectionMethod == "tournament" and self.participants > self.popSize:
			print("Error: invalid settings file.\nMore participants in a tournament than individuals in the population.")
			sys.exit()

		if self.mutationSelect not in ["prev gen", "gene pool"]:
			print("Error: invalid settings file.\nNeed to provide a valid population from which to generate mutants.")	
			sys.exit()

		if self.crossoverR + self.cloneR > 1:
			print("Error: invalid setting file.\nThe crossover rate and the clone rate cannot exceed 1.")

		if Individual.stray not in ["nothing", "para to boundary", "para randomise", "ind randomise"]:
			print("Error: invalid setting file.\nDo not know what to do when an individual strays beyond the boundary.")
			# sys.exit()
		###
		# Calculate all the other variables
		###

		# number of individuals produced by crossover, rounded up
		self.crossoverN = int(np.ceil(self.crossoverR * self.popSize))
		# if offspring are produced in groups of self.surviveN, there are self.crossoverN / self.surviveN litters
		# eg 3 offspring produced from a pair of parents. For a population of 10, need 4 litters
		#	(round up because you want to produce excess offspring)
		# hence, need twice as many parents
		# at the end we just delete randomly so that in fact some parents had a slightly lower
		# offspring survival rate
		# must always be even
		self.genePoolN = 2 * int(np.ceil(self.crossoverN / self.surviveN))

		# when ranking, the individual with the better fitness score comes first
		# if is a minimisation problem, we can just sort
		# if is a maximisation problem, we need to sort in reverse order
		if self.problemType == "minimise":
			self.reverse = False
		else:
			self.reverse = True

		self.cloneN = int(np.ceil(self.cloneR * self.popSize))

		# because of the rounding, the number of individuals produced by crossover and cloning may
		# exceed the population size, but only by 1
		if self.cloneN + self.crossoverN > self.popSize:
			self.crossoverN -= 1

		# How many mutants do we want to make?
		self.mutateN = self.popSize - self.cloneN - self.crossoverN

		if self.crossoverN + self.cloneN + self.mutateN != self.popSize:
			print("Error: crossoverN, cloneN and mutateN do not add to popSize")
			sys.exit()

		self.cullN = int(np.floor(self.cullR * self.popSize))

		self.simInfo = """===== Simulation Information =====
population size: {popSize}
produced by crossover: {crossoverN}
produced by cloning: {cloneN}
produced by mutation: {mutateN}
number of individuals culled: {cullN}""".format(
	popSize = self.popSize, 
	crossoverN = self.crossoverN, 
	cloneN = self.cloneN, 
	mutateN = self.mutateN, 
	cullN = self.cullN)
		print(self.simInfo)
		
		###
		# Initialising the results
		###

		# clear the results directory
		subprocess.call(["rm", "-r", self.resFolder])
		subprocess.call(["mkdir", self.resFolder])
		subprocess.call(["cp", settings, self.resFolder + "/settings.txt"])
		subprocess.call(["mkdir", self.resFolder + "/popAtEachGen"])
		subprocess.call(["mkdir", self.resFolder + "/allIndividuals"])

		if self.saveData == True:
			subprocess.call(["mkdir", self.resFolder + "/data"])

		f = open(self.resFolder + "/simInfo.txt", "w")
		f.write(self.simInfo)
		f.close()

		self.aveFitScore = []	# average fitness score at each generation
		self.bestInd = []		# stats of best individual (paraVals and fitness score) at each generation
		self.bestFit = []		# best fitness score at each generation
		###
		# Create initial population
		###

		self.nextGen = []
		self.currentEpoch = '0'
		Individual.allInd = []
		print('=' * 10 + "Creating initial population" + '=' * 10)
		for i in range(self.popSize):
			# initialse parameters uniformly distributed
			paraVals = [np.random.uniform(self.lBound[p], self.uBound[p]) for p in range(self.paraN)]
			ind = Individual(paraVals)
			self.nextGen.append(ind)
			print(ind.paraVals, ind.fitScore)
		self.refresh()
		
		###
		# Run simulation
		###

		self.nextGen = []
		i = 1	# counts the number of epochs
		while self.epochs != 0:
		# for i in range(1, self.epochs + 1):
			self.currentEpoch = str(i)
			print('=' * 10 + 'Starting epoch: ' + self.currentEpoch + '=' * 10)
			self.cull()
			self.select()
			self.crossover()
			self.mutate()
			self.clone()
			
			for j in self.nextGen:
				print(j.paraVals, j.fitScore, j.time)
			self.refresh()
			i += 1

		# write the new number of epochs at the bottom of the file so the results know how many epochs
		f = open(self.resFolder + "/settings.txt", "a")
		f.write("total epochs," + str(self.currentEpoch))
		f.close()
	#################### FUNCTIONS ####################

	def refresh(self):
		''' deletes the old generation and overwites it with the new generation 
		also does all the saving'''

		self.prevGen = self.nextGen
		self.nextGen = []

		# this is the population that exists at the start of epoch. So it includes those that are culled, but
		# it doesnt include individuals that don't get born - i.e. dont survive during crossover
		population = [ind.stats for ind in self.prevGen]
		np.save(self.resFolder + "/popAtEachGen/pop_" + self.currentEpoch + ".npy", population)
		np.save(self.resFolder + "/allIndividuals/ind_" + self.currentEpoch + ".npy",Individual.allInd)
		print('++++')
		for i in Individual.allInd:
			print(i)
		print('++++')
		Individual.allInd = []

		# calculate the average fitness:
		aveFS = np.average([ind.fitScore for ind in self.prevGen])
		self.aveFitScore.append(aveFS)
		np.save(self.resFolder + "/aveFitScore.npy", self.aveFitScore)
		print(self.aveFitScore)

		# save the best individual:
		ranked = sorted(self.prevGen, key = attrgetter("fitScore"), reverse = self.reverse)
		self.bestInd.append(ranked[0].stats)
		self.bestFit.append(ranked[0].fitScore)
		np.save(self.resFolder + "/bestInd.npy", self.bestInd)
		if self.saveData == True:
			np.save(self.resFolder + "/data/" + "data_" + str(self.currentEpoch), ranked[0].data)
		
		# check stopping criteria: best
		# if we have fulfilled the minimum number of epochs and the change in the best fitness
		# score is still too large, keep going

		if self.stoppingCriteria == "best" and self.epochs == 1 and abs(self.bestFit[-1] - self.bestFit[0]) > self.epsilon:
			self.bestFit = self.bestFit[1:]	# knock off the first one
		else:
			self.epochs -= 1

	###
	# Culling:
	# 	eliminate self.cullN worst individuals
	###

	def cull(self):
		if self.cullN != 0:
			# rank the previous population (best at the front) - chop off the bad tail
			self.prevGen = sorted(self.prevGen, key = attrgetter("fitScore"), reverse = self.reverse)[:-self.cullN]

	###
	# Selection:
	# 	Each selection method creates self.genePool through selection from self.prevGen which 
	#	is a list of individuals that will go on to reproduce. The number of individuals is 
	#	specified by self.genePoolN
	###

	def tournament(self):
		''' Essentially the best participant has a win rate chance of entering the gene pool. 
		If that fails, the next best gets a win rate chance of entering the gene pool. 
		This continues and if no individual wins, the least fit indiviual enters the gene pool. 
		Note an individual cannot participate more than once in a tournament. '''
		self.genePool = []	# create an empty gene pool

		# 	each tournament adds one individual to the gene pool
		for tournament in range(self.genePoolN):
			participants = np.random.choice(self.prevGen, size = self.participants, replace = False)
			# sort them by rank
			participants = sorted(participants, key = attrgetter("fitScore"), reverse = self.reverse)

			# iterate through the list of the participants, starting from the best. 
			# At each iteration there is a self.winRate chance of the individual 'winning'
			# if there are no winners, the worst participant is added to the genePool
			winner = participants[-1]
			for ind in participants:
				if np.random.uniform() < self.winRate:
					winner = ind
					break
			self.genePool.append(winner)

	def rouletteWheel(self):
		self.genePool = []

		# obtain area on the roulette wheel
		if self.problemType == "minimise":
			# need to invert the fitness score
			maxFitScore = np.max([i.fitScore for i in self.prevGen])
			area = [maxFitScore - i.fitScore + 1 for i in self.prevGen]
		else:
			# dont need to invert fitness score - area is proportional to the fit score
			area = [i.fitScore for i in self.prevGen]

		# set up an accumulative distribution
		accDist = list(itertools.accumulate(area))

		if self.markerDist == "SUS":
			markers = np.linspace(0, max(accDist), self.genePoolN)
		else:
			markers = np.random.uniform(0, max(accDist), self.genePoolN)

		for m in markers:
			index = sum([1 for i in accDist if i < m])
			self.genePool.append(self.prevGen[index])

		shuffle(self.genePool)


	###
	# Crossover:
	# 	Each crossover method uses individuals from self.genePool and adds self.crossoverN 
	#	individuals to self.nextGen. Note that self.genePoolN is always even
	###

	def multiPoint(self):

		producedOffspring = []	# all the offspring produced (will sometimes make excess)
		parentsN = int(0.5*self.genePoolN)	# number of parent pairs
		np.random.shuffle(self.genePool)
		mothers = self.genePool[:parentsN]
		fathers = self.genePool[parentsN:]

		# for each set of parents
		for i in range(parentsN):
			m = mothers[i]
			f = fathers[i]

			offspring = []	# offspring prduced by the parents
			for ind in range(self.broodN):
				# size: no of parameters its picks. at least 1 parameter but less than self.paraN parameters
				# chooses randomly without replacement which parameters to put in the mask
				mask = np.random.choice(range(self.paraN), size = np.random.randint(1, self.paraN), replace = False)
				
				# create the offspring
				offspringVals = []
				for para in range(self.paraN):
					if para in mask:
						offspringVals.append(m.paraVals[para])
					else:
						offspringVals.append(f.paraVals[para])
				offspring.append(Individual(offspringVals))

			# order the offspring so that the best are first
			offspring = sorted(offspring, key = attrgetter("fitScore"), reverse = self.reverse)

			# only self.surviveN actually survive
			for ind in range(self.surviveN):
				producedOffspring.append(offspring[ind])

		# Sometimes we make excess children. only want to add self.crossoverN children do the next gen
		# make a random selection to add
		for i in np.random.choice(producedOffspring, size = self.crossoverN, replace = False):
			self.nextGen.append(i)

	###
	# Mutation:
	# 	selects self.mutateN from either the previous generation or the gene pool to make up 
	#	'originals' and mutates all of them and adds the mutants to self.nextGen. Note an
	# 	individual can spawn multiple mutants
	###

	def randomMutation(self):
		''' This is the random mutation function described in Tang, Tseng, 2012 '''
		if self.mutationSelect == 'gene pool':
			self.originals = self.genePool
		else:
			self.originals = self.prevGen
		# make a selection of 'originals' to mutate
		self.originals = np.random.choice(self.originals, size = self.mutateN, replace = True)
		
		# work out the ranges in all the parameters in the current generation
		paraMatrix = np.matrix([i.paraVals for i in self.prevGen])
		paraMax = np.array(np.amax(paraMatrix, axis = 0))[0]
		paraMin = np.array(np.amin(paraMatrix, axis = 0))[0]

		for o in self.originals:
			mutantVals = []
			for para in range(self.paraN):
				delta = max(2*(o.paraVals[para] - paraMin[para]), 2*(paraMax[para] - o.paraVals[para]))
				mutantVals.append(o.paraVals[para] + delta * (np.random.uniform(0, 1) - 0.5))
			mutant = Individual(mutantVals)
			self.nextGen.append(mutant)


	###
	# Clone:
	# 	selects self.cloneN best individuals and adds them to the next generation
	###

	def clone(self):

		# rank the previous population
		ranked = sorted(self.prevGen, key = attrgetter("fitScore"), reverse = self.reverse)

		for i in range(self.cloneN):
			self.nextGen.append(ranked[i])

class Results:
	''' Note that this results in completely independent of the GA, and reads in everything
	from the results folder'''

	def __init__(self, resFolder):
		
		self.resFolder = resFolder
		self.settings = resFolder + "/settings.txt"
		self.info = resFolder + "/simInfo.txt"

		f = open(self.settings)
		for line in f:
			line = line.strip().split(',')

			if line[0] == "total epochs":
				self.epochs = int(line[1].strip())

			if line[0] == "upper bounds":
				self.uBound = [float(ub.strip()) for ub in line[1:]]
			if line[0] == "lower bounds":
				self.lBound = [float(lb.strip()) for lb in line[1:]]

			if line[0] == "parameter list":
				self.paraName = [para.strip() for para in line[1:]]
				self.paraN = len(self.paraName)
				Individual.paraName = self.paraName
				Individual.paraN = self.paraN
			if line[0] == "save data":
				if line[1].strip() == 'T':
					self.saveData = True
		f.close()

		f = open(self.info)
		self.simInfo = f.read()
		f.close()
		print(self.simInfo)

		self.generations = [i for i in range(self.epochs + 1)]

		# read in the stats of the best results at each generation
		self.bestInd = np.load(self.resFolder + "/bestInd.npy")

		# read in the data of the best results at each generation
		self.bestIndData = [np.load(self.resFolder + "/data/data_" + str(i) + ".npy") for i in range(self.epochs + 1)]

		self.bestFitScore = [i[-2] for i in self.bestInd]

		# read in average fitness score at each generation
		self.aveFitScore = np.load(self.resFolder + "/aveFitScore.npy")

		# read in the populations
		self.population = [np.load(self.resFolder + "/popAtEachGen/pop_" + str(i) + ".npy") for i in range(0, self.epochs + 1)]

		# read in all models made at each generation stored by generation
		self.allIndGen = [np.load(self.resFolder + "/allIndividuals/ind_" + str(i) + ".npy") for i in range(0, self.epochs + 1)]

		# read in all the models made ever
		self.allInd = [ind for gen in self.allIndGen for ind in gen]


	def help(self):
		helpMessage = """ ===== The Results class =====
----- attributes -----
The Results class has a number of useful attributes:
All 'individuals' are represented by a list. They are the parameter values
followed by the fitness score.

bestInd: gives you the stats of the best individual in each generation.
There is the best individual for each epoch and the best individual in
initial population.

bestIndData: gives the data of the best individuals in each generation. 
The data has to be an array of values and is defined in the evaluation
function by the user. Only saves if 'save data' is T in the settings
file.

aveFitScore: an array of the best fitness score at each generation.

population: is a 2D array. It stores the population at the end of
each epoch (before culling takes place) but it doesn't include the
individuals that aren't born, i.e. don't survive crossover. The first
array is the initial population.

allIndGen: is a 2D array of all the individuals created during that epoch.
Inlcudes all individuals, i.e. even the individuals that weren't 'born' i.e.
did not survive crossover.

allInd: is a 1D array of allIndGen and contains all the individuals ever created.

----- methods -----
Some basic visualisation functions have been provided for convenience. For 
more advanced visualisations, users should write their own.

AveFitScore: plots the average fitness score of the population at each 
generation. (includes individuals that will be culled but not individuals 
that weren't born).

bestFitScore: plots the best fitness score in the population at each generation.

fitScores: plots both the average fitness scores and best fitness scores on the 
same plot.

errorspace3d: plots every individual that existed. The colour represents
their fitness score.

"""

		print(helpMessage)

	def aveFitScores(self, ind = -1, small_ave = 0.1):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.plot(self.generations[:ind], self.aveFitScore[:ind])
		plt.xlabel("Epoch")
		plt.ylabel("Fitness Score")
		plt.title("Average fitness score at each epoch")
		ax.set_xlim([min(self.generations), max(self.generations)])
		ax.set_ylim([min(self.aveFitScore), max(self.aveFitScore)])
		plt.savefig(self.resFolder + "/aveFitScores/"+str(ind)+".png")
		plt.close()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.plot(self.generations[:ind], self.aveFitScore[:ind])
		plt.xlabel("Epoch")
		plt.ylabel("Fitness Score")
		plt.title("Average fitness score at each epoch")
		ax.set_xlim([min(self.generations), max(self.generations)])
		ax.set_ylim([min(self.aveFitScore), small_ave*max(self.aveFitScore)])
		plt.savefig(self.resFolder + "/aveFitScores_small/"+str(ind)+".png")
		plt.close()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.plot(self.generations[:ind], np.log10(self.aveFitScore[:ind]))
		plt.xlabel("Epoch")
		plt.ylabel("Fitness Score (log 10)")
		plt.title("Average fitness score at each epoch")
		ax.set_xlim([min(self.generations), max(self.generations)])
		ax.set_ylim([np.log10(min(self.aveFitScore)), np.log10(max(self.aveFitScore))])
		plt.savefig(self.resFolder + "/aveFitScores_log/"+str(ind)+".png")
		plt.close()

	def bestFitScores(self, ind = -1, small_best = 0.1):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.plot(self.generations[:ind], self.bestFitScore[:ind])
		plt.xlabel("Epoch")
		plt.ylabel("Fitness Score")
		plt.title("Best fitness score at each epoch")
		ax.set_xlim([min(self.generations), max(self.generations)])
		ax.set_ylim([min(self.bestFitScore), max(self.bestFitScore)])
		plt.savefig(self.resFolder + "/bestFitScores/"+str(ind)+".png")
		plt.close()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.plot(self.generations[:ind], self.bestFitScore[:ind])
		plt.xlabel("Epoch")
		plt.ylabel("Fitness Score")
		plt.title("Best fitness score at each epoch")
		ax.set_xlim([min(self.generations), max(self.generations)])
		ax.set_ylim([min(self.bestFitScore), small_best*max(self.bestFitScore)])
		plt.savefig(self.resFolder + "/bestFitScores_small/"+str(ind)+".png")
		plt.close()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.plot(self.generations[:ind], np.log10(self.bestFitScore[:ind]))
		plt.xlabel("Epoch")
		plt.ylabel("Fitness Score (log 10)")
		plt.title("Best fitness score at each epoch")
		ax.set_xlim([min(self.generations), max(self.generations)])
		ax.set_ylim([np.log10(min(self.bestFitScore)), np.log10(max(self.bestFitScore))])
		plt.savefig(self.resFolder + "/bestFitScores_log/"+str(ind)+".png")
		plt.close()

	def fitScores(self, ind = -1, small_ave = 0.1, small_best = 0.1):

		fig, ax1 = plt.subplots(figsize = (6, 6))
		ax1.plot(self.generations[:ind], self.bestFitScore[:ind], 'b')
		ax1.set_ylabel("Best fitness score at each epoch", color = 'b')
		ax1.set_xlabel("Epoch")
		ax1.tick_params('y', colors = 'b')
		ax1.set_xlim([min(self.generations), max(self.generations)])
		ax1.set_ylim([min(self.bestFitScore), max(self.bestFitScore)])

		ax2 = ax1.twinx()
		ax2.plot(self.generations[:ind], self.aveFitScore[:ind], 'g')
		ax2.set_ylabel("Average fitness score at each epoch", color = 'g')
		ax2.tick_params('y', colors = 'g')
		ax2.set_xlim([min(self.generations), max(self.generations)])
		ax2.set_ylim([min(self.aveFitScore), max(self.aveFitScore)])

		plt.title("Fitness score at each epoch")
		plt.savefig(self.resFolder + "/fitScores/"+str(ind)+".png")
		plt.close()

		fig, ax1 = plt.subplots(figsize = (6, 6))
		ax1.plot(self.generations[:ind], self.bestFitScore[:ind], 'b')
		ax1.set_ylabel("Best fitness score at each epoch", color = 'b')
		ax1.set_xlabel("Epoch")
		ax1.tick_params('y', colors = 'b')
		ax1.set_xlim([min(self.generations), max(self.generations)])
		ax1.set_ylim([min(self.bestFitScore), small_best*max(self.bestFitScore)])

		ax2 = ax1.twinx()
		ax2.plot(self.generations[:ind], self.aveFitScore[:ind], 'g')
		ax2.set_ylabel("Average fitness score at each epoch", color = 'g')
		ax2.tick_params('y', colors = 'g')
		ax2.set_xlim([min(self.generations), max(self.generations)])
		ax2.set_ylim([min(self.aveFitScore), small_ave*max(self.aveFitScore)])

		plt.title("Fitness score at each epoch")
		plt.savefig(self.resFolder + "/fitScores_small/"+str(ind)+".png")
		plt.close()

		fig, ax1 = plt.subplots(figsize = (6, 6))
		ax1.plot(self.generations[:ind], np.log10(self.bestFitScore[:ind]), 'b')
		ax1.set_ylabel("Best fitness score at each epoch (log 10)", color = 'b')
		ax1.set_xlabel("Epoch")
		ax1.tick_params('y', colors = 'b')
		ax1.set_xlim([min(self.generations), max(self.generations)])
		ax1.set_ylim([np.log10(min(self.bestFitScore)), np.log10(max(self.bestFitScore))])

		ax2 = ax1.twinx()
		ax2.plot(self.generations[:ind], np.log10(self.aveFitScore[:ind]), 'g')
		ax2.set_ylabel("Average fitness score at each epoch (log 10)", color = 'g')
		ax2.tick_params('y', colors = 'g')
		ax2.set_xlim([min(self.generations), max(self.generations)])
		ax2.set_ylim([np.log10(min(self.aveFitScore)), np.log10(max(self.aveFitScore))])

		plt.title("Fitness score at each epoch")
		plt.savefig(self.resFolder + "/fitScores_log/"+str(ind)+".png")
		plt.close()

	def errorspace3d(self, p1 = 0, p2 = 1, p3 = 2, maxPercentile = 20, value = None, colormap = cm.rainbow, d = False):

		'''p1, p2 and p3 are the index of the parameters as specified in the
		settings file '''

		fig = plt.figure(figsize = (8,6))
		ax = fig.add_subplot(111, projection = "3d")
		if d == True:
			para1 = [i[p1]*2 for i in self.allInd]
			para2 = [i[p2]*2 for i in self.allInd]
			para3 = [i[p3] for i in self.allInd]
		else:
			para1 = [i[p1] for i in self.allInd]
			para2 = [i[p2] for i in self.allInd]
			para3 = [i[p3] for i in self.allInd]

		colors = [i[-2] for i in self.allInd]	# all the errors
		pColors = []
		for i in colors:
			if value != None:
				val = value
			else:
				val = np.percentile(colors, maxPercentile)
			if i <= val:
				pColors.append(i)
			else:
				pColors.append(val)

		scatPlt = ax.scatter(para1, para2, para3, c = pColors, cmap = colormap, marker = 'x')
		plt.ticklabel_format(axis = 'z', scilimits = (0,0))

		ax.set_xlabel(self.paraName[p1])
		ax.set_ylabel(self.paraName[p2])
		ax.set_zlabel(self.paraName[p3])

		cb = plt.colorbar(scatPlt)
		cb.set_label("Fitness score")

		plt.title("Error space")

	def errorspace2d(self, p1 = 0, p2 = 1, maxPercentile = 20, value = None, colormap = cm.rainbow):

		fig = plt.figure()
		ax = fig.add_subplot(111)

		para1 = [i[p1] for i in self.allInd]
		para2 = [i[p2] for i in self.allInd]
		colors = [i[-1] for i in self.allInd]
		pColors = []
		for i in colors:
			if value != None:
				val = value
			else:
				val = np.percentile(colors, maxPercentile)
			if i <= val:
				pColors.append(i)
			else:
				pColors.append(val)

		scatPlt = plt.scatter(para1, para2, marker = 'x', c = pColors)
		plt.xlabel(self.paraName[p1])
		plt.ylabel(self.paraName[p2])
		ax.set_xlim(min(para1), max(para1))

		cb = plt.colorbar(scatPlt)
		cb.set_label("Fitness score")

		plt.title("Error space")
		plt.show()


	def exploration3d(self, p1 = 0, p2 = 1, p3 = 2, start = 0, analytic = None, gens = None, scale = True, d = False):

		colors = iter(cm.rainbow(np.linspace(0, 1, self.epochs + 1-start)))
		if d == True:
			para1 = [i[p1]*2 for i in self.allInd]
			para2 = [i[p2]*2 for i in self.allInd]
			para3 = [i[p3] for i in self.allInd]
		else:
			para1 = [i[p1] for i in self.allInd]
			para2 = [i[p2] for i in self.allInd]
			para3 = [i[p3] for i in self.allInd]
		print(gens)
		fig = plt.figure(figsize = (8,7))
		ax1 = plt.subplot2grid((1, 15), (0, 0), colspan = 14, projection = '3d')
		for i in range(start, self.epochs):
			if d == True:
				para1 = [ind[p1]*2 for ind in self.allIndGen[i]]
				para2 = [ind[p2]*2 for ind in self.allIndGen[i]]
				para3 = [ind[p3] for ind in self.allIndGen[i]]
			else:
				para1 = [ind[p1] for ind in self.allIndGen[i]]
				para2 = [ind[p2] for ind in self.allIndGen[i]]
				para3 = [ind[p3] for ind in self.allIndGen[i]]
			if gens == None or i in gens:
				scatPlt = ax1.scatter(para1, para2, para3, marker = 'o', c = next(colors), cmap = cm.rainbow, alpha = 0.5)
			else:
				next(colors)

		if analytic != None:
			scatPlt = ax1.scatter(analytic[0], analytic[1], analytic[2], marker = '*', c = 'black', s = 50)
		ax1.set_xlabel(self.paraName[p1])
		ax1.set_ylabel(self.paraName[p2])
		ax1.set_zlabel(self.paraName[p3])
		if scale:
			# keep the axes consistent
			if d == True:
				ax1.set_xlim([self.lBound[p1]*2, self.uBound[p1]*2])
				ax1.set_ylim([self.lBound[p2]*2, self.uBound[p2]*2])
				ax1.set_zlim([self.lBound[p3], self.uBound[p3]])
			else:
				ax1.set_xlim([self.lBound[p1], self.uBound[p1]])
				ax1.set_ylim([self.lBound[p2], self.uBound[p2]])
				ax1.set_zlim([self.lBound[p3], self.uBound[p3]])
		plt.ticklabel_format(style='sci', axis='z', scilimits=(0,0))

		plt.title("Progression of the population")
		ax2 = plt.subplot2grid((1, 15), (0, 14))
		cmap = cm.rainbow
		norm = mpl.colors.Normalize(vmin=start, vmax = self.epochs + 1)

		cb1 = mpl.colorbar.ColorbarBase(ax2, cmap = cmap, norm = norm, orientation = 'vertical')
		cb1.set_label('Generation')

		plt.show()

	def exploration3d_animate(self, p1 = 0, p2 = 1, p3 = 2, start = 0, analytic = None, d = False):

		colors = iter(cm.rainbow(np.linspace(0, 1, self.epochs + 1-start)))

		if d == True:
			para1 = [i[p1]*2 for i in self.allInd]
			para2 = [i[p2]*2 for i in self.allInd]
			para3 = [i[p3] for i in self.allInd]
		else:
			para1 = [i[p1] for i in self.allInd]
			para2 = [i[p2] for i in self.allInd]
			para3 = [i[p3] for i in self.allInd]

		for i in range(start, self.epochs + 1):
			fig = plt.figure(figsize = (7,6))
			ax1 = plt.subplot2grid((1, 15), (0, 0), colspan = 14, projection = '3d')
			para1 = [ind[p1] for ind in self.allIndGen[i]]
			para2 = [ind[p2] for ind in self.allIndGen[i]]
			para3 = [ind[p3] for ind in self.allIndGen[i]]
			scatPlt = ax1.scatter(para1, para2, para3, marker = 'o', c = next(colors), cmap = cm.rainbow)
			if analytic != None:
				scatPlt = ax1.scatter(analytic[0], analytic[1], analytic[2], marker = '*', c = 'black', s = 50)
			ax1.set_xlabel(self.paraName[p1])
			ax1.set_ylabel(self.paraName[p2])
			ax1.set_zlabel(self.paraName[p3])
			ax1.set_xlim([self.lBound[p1], self.uBound[p1]])
			ax1.set_ylim([self.lBound[p2], self.uBound[p2]])
			ax1.set_zlim([self.lBound[p3], self.uBound[p3]])
			plt.title("Progression of the population: gen " + str(i))
			ax2 = plt.subplot2grid((1, 15), (0, 14))
			cmap = cm.rainbow
			norm = mpl.colors.Normalize(vmin=start, vmax = self.epochs + 1)

			cb1 = mpl.colorbar.ColorbarBase(ax2, cmap = cmap, norm = norm, orientation = 'vertical')
			cb1.set_label('Generation')
			plt.savefig(self.resFolder + "/progression/"+str(i)+".png")
			plt.close()




