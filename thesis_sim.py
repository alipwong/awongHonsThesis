from readDiffdata import *
from phys_helper import *
from GA_classes import *

from results_class import *

def evaluate(individual):
	'''takes in an individual defined by parameters and outputs a fitness score
		individual.paraName = [<list of parameter names]
		individual.paraVal = [<list of parameter values]
		=> returns
		individual.fitScore
		optional: individual.data'''
	
	delete = True

	# set the parameter attributes so that a parameter file can be written
	for i in range(individual.paraN):
		individual.__dict__[individual.paraName[i]] = individual.paraVals[i]

	# check that the stellar radius is less than the inner radius of the dust shell
	# if its too small, expand the inner radius
	alpha = 0.015	# buffer, rin greater than radius + alpha
	offset = np.sqrt(pow(individual.x, 2) + pow(individual.y, 2) + pow(individual.z, 2))

	if np.round(individual.rin, decimals = 1) <= np.round(sr_to_AU(individual.radius) + offset, decimals = 1):
		individual.rin = np.round(sr_to_AU(individual.radius) + offset, decimals = 5) + alpha

	# Delete old data
	delete_old(individual.wavelength, individual.mcfostPath)	
	write_para(individual)
	
	individual.modelData, individual.time = make_model(individual) # disp_mcfost = True

	# obtain the images
	individual.I, individual.Q, individual.U = load_fits(individual.modelData)
	individual.H, individual.V, individual.A, individual.B = get_polarised_imgs(individual.I, individual.Q, individual.U)

	# get the normalised and centered power spectrums
	individual.Hps = normalise(center(ps(individual.H)))
	individual.Vps = normalise(center(ps(individual.V)))
	individual.Aps = normalise(center(ps(individual.A)))
	individual.Bps = normalise(center(ps(individual.B)))

	if delete:
		delattr(individual, 'I')
		delattr(individual, 'Q')
		delattr(individual, 'U')
		delattr(individual, 'H')
		delattr(individual, 'V')
		delattr(individual, 'A')
		delattr(individual, 'B')

	individual.visHV = individual.Hps/individual.Vps
	individual.visAB = individual.Aps/individual.Bps

	if delete:
		delattr(individual, 'Hps')
		delattr(individual, 'Vps')
		delattr(individual, 'Aps')
		delattr(individual, 'Bps')

	#--- from Barnaby
	individual.nyqval = individual.masPerPix * 2
	individual.nyqval_rad = mas_to_rad(individual.nyqval)
	individual.maxBLlength = individual.wavelength_m / individual.nyqval_rad

	# assuming u, v aex are symmetric
	individual.U_vals = np.linspace(-1, 1, individual.grid) * individual.maxBLlength
	individual.V_vals = individual.U_vals
	#---

	# make measurements
	individual.sampledVisHV = make_measurements(individual, individual.visHV)
	individual.sampledVisAB = make_measurements(individual, individual.visAB)
	if delete:
		delattr(individual, 'visHV')
		delattr(individual, 'visAB')

	# calculate the chi^2 error
	individual.chi2errHV = np.sum(pow((individual.obsData.vhvv - individual.sampledVisHV)/individual.obsData.vhvverr,2))
	individual.chi2errAB = np.sum(pow((individual.obsData.vhvvu - individual.sampledVisAB)/individual.obsData.vhvvuerr,2))
	individual.chi2redHV = individual.chi2errHV/(individual.n - individual.paraN)
	individual.chi2redAB = individual.chi2errAB/(individual.n - individual.paraN)
	
	individual.fitScore = 0.5*(individual.chi2redHV + individual.chi2redAB)
	individual.data = [individual.sampledVisHV, individual.sampledVisAB]

def initIndividual(evaluate, obsData):

	Individual.evaluate = evaluate
	Individual.obsData = obsData

	# Set other values:
	Individual.mcfostPath = "./"
	Individual.parentPara = Individual.mcfostPath + "RLeo.para"
	Individual.modelPara = Individual.mcfostPath + "modified2.para"

	Individual.wavelength = 0.75
	Individual.grid = 1001
	Individual.size = 200
	Individual.distance = 100
	Individual.mass = 1.0
	Individual.imageSym = 'T'
	Individual.centralSym = 'T'
	Individual.axialSym = 'T'
	Individual.x = 0
	Individual.y = 0
	Individual.z = 0

	# parameters in other units
	Individual.wavelength_m = mum_to_m(Individual.wavelength)
	Individual.size_m = AU_to_m(Individual.size)
	Individual.distance_m = pc_to_m(Individual.distance)

	# Load in constants from the data file
	Individual.uMeters = Individual.obsData.u_coords * Individual.wavelength_m
	Individual.vMeters = Individual.obsData.v_coords * Individual.wavelength_m
	Individual.angles = Individual.obsData.bazims
	Individual.blengths = Individual.obsData.blengths

	# Calculation of other constants
	Individual.masPerPix = calculate_masPerPx(Individual.grid, Individual.size_m, Individual.distance_m)

	# counts
	Individual.n = len(Individual.uMeters)	# number of sampled points
	Individual.allInd = []
###
#	Running mechanisms
###

howToRun = """specify a settings file, and state whether you want to run the GA or load results
To run a GA:
python3 <this>.py <settings file>

To load results:
python3 <this>.py -load <results folder> 
"""

# reading in the observed data
# --- from Barnaby
srcPath = "./mcfost_sim/"
srcCubeInfoFilename = 'cubeinfoMar2017.idlvar'
srcFilename = 'diffdata_RLeo_03_20170313_750-50_18holeNudged_0_0.idlvar'
obsData = vampDiffdata(srcPath+srcFilename, srcPath+srcCubeInfoFilename)

# first order correction
correction = True
if correction:
	bias_vhvv = np.mean(obsData.vhvv) - 1
	bias_vhvvu = np.mean(obsData.vhvvu) - 1

	obsData.vhvv = obsData.vhvv - bias_vhvv
	obsData.vhvvu = obsData.vhvvu - bias_vhvvu

# running simulation
if len(sys.argv) == 2:

	# check it can find the settings file
	settings = sys.argv[1]
	if os.path.isfile(settings) == False:
		print("Error: cannot find settings file")
		sys.exit()

###
#	Run simulation: do all the setup in here
###

	initIndividual(evaluate, obsData)
	sim = GA(settings)

# load a file
elif len(sys.argv) == 3 and sys.argv[1] == "-load":

	# check to see if results file exists
	resFolder = sys.argv[2]
	Individual.resFolder = resFolder

	if os.path.isfile(resFolder) == "False":
		print("Error: cannot find results file")
		sys.exit()
	
###
#	Write your code to display results in here
###
	
	print("loading results...")

	initIndividual(evaluate, obsData)
	Results.epochs = 100
	results = Results(resFolder)
	n_ind = len(results.allInd) 
	print(len(results.allInd))
	n_epochs = len(results.population)
	print(len(results.population))

	results.errorspace3d(p1 = 2, p2 = 1, p3 = 0, value = 50)
	results.exploration3d(p1 = 2, p2 = 1, p3 = 0)
	results.exploration3d(p1 = 2, p2 = 1, p3 = 0, scale = False, gens = list(range(results.epochs-9,results.epochs)))

	results.exploration3d(p1 = 3, p2 = 4, p3 = 5, analytic = [1.5, -1, 0.8])
	results.exploration3d(p1 = 3, p2 = 4, p3 = 5, analytic = [1.5, -1, 0.8], scale = False, gens = list(range(70,88)))
	results.errorspace3d(p1 = 3, p2 = 4, p3 = 5, value = 50)

	# calculate the time

	for i in results.bestInd:
		print(i)

	times = []
	for i in results.allInd:
		times.append(i[-1])
	print(max(times))
	print("totaltime (s):", sum(times))
	print("totaltime (h):", sum(times)/60/60)
	print("ave time (s):", sum(times)/n_ind)
	print("ave time (h):", sum(times)/n_ind/60/60)

	# automation to produce animations
	folders = ["/Q_animate", "/U_animate", "/correlation", "/progression", "/fitScores_small", "/bestFitScores_small", "/aveFitScores_small", "/fitScores", "/bestFitScores", "/aveFitScores", "/fitScores_log", "/bestFitScores_log", "/aveFitScores_log"]

	for i in folders:
		subprocess.call(['rm', '-r', Individual.resFolder + i])
		subprocess.call(['mkdir', Individual.resFolder + i])

	for e in range(results.epochs + 1):
		# plot the data:
		results.aveFitScores(ind = e, small_ave = 1E-10)
		results.bestFitScores(ind = e, small_best = 0.01)
		results.fitScores(ind = e, small_ave = 1E-10, small_best = 0.01)
		correlation(e, Individual, results.bestIndData[e][0], results.bestIndData[e][1])
		plot_vis2d(e, Individual, results.bestIndData[e][0], stokes = 'Q')
		plot_vis2d(e, Individual, results.bestIndData[e][1], stokes = 'U')

	subprocess.call(['mkdir', Individual.resFolder + "/progression"])
	results.exploration3d_animate(p1 = 0, p2 = 1, p3 = 2, d = double)
#analytic = [3.5E-6, 2, 150]

else:
	print(howToRun)
	sys.exit()


