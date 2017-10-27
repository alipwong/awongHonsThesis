from scipy.fftpack import fft2, ifft2
from astropy.io import fits
from scipy import interpolate
import numpy as np
import subprocess
import time

import matplotlib.pyplot as plt
#--------------
#	Unit conversions
#--------------

def pc_to_m(x):
	''' converts from parsecs to meters'''
	return x * 3.08567758149137e16

def AU_to_m(x):
	''' converts from AU to meters'''
	return x * 1.49597870700e11

def rad_to_mas(x):
	''' converts from radians to milliarcseconds'''
	return x / np.pi * 180 * 60 * 60 * 1000

def mas_to_rad(x):
	''' converts from radians to milliarcseconds'''
	return x * np.pi / 180 / 60 / 60 / 1000

def mum_to_m(x):
	''' coverts from micrometers to meters'''
	return x*1e-6

def rad_to_deg(x):
	''' converts from radians to degrees'''
	return x / np.pi * 180

def sr_to_AU(x):
	''' converts from solar radius to AU'''
	return x*0.00465047

#--------------
#	Get constants
#--------------

def angular_size(size_m, dist_m):
	''' takes in a physical size x in meters and the distance to the object 
	in meters and returns the angular size in milliarcseconds'''
	return rad_to_mas(np.arctan(size_m/dist_m))

def calculate_masPerPx(grid, imgSz_m, dist_m):
	''' given the number of pixels across, the size of that in meters and the distance to the object,
	calculate the angular size of a pixel'''
	mPerPx = imgSz_m / grid
	masPerPx = angular_size(mPerPx, dist_m)

	return masPerPx
def load_fits(data):
	I = data[0][0][0]
	Q = data[1][0][0]
	U = data[2][0][0]

	# can turn these on if required
	#P = np.sqrt(np.square(Q) + np.square(U))	# polarised intensity
	#p = P/I # fractional polarisation - degree of polarisation
	return I, Q, U

def get_polarised_imgs(I, Q, U):
	# swapping coords
	V = I + Q
	H = I - Q
	B = I + U
	A = I - U
	return H, V, A, B

#--------------
#	Image Manipulation
#--------------

def center(image):
	""" Re-centers an image from the origin to the middle of the display image"""
	size = image.shape
	half = int(np.ceil(size[0]/2))
	image = np.roll(np.roll(image, half, 0), half, 1)
	return image

def ps(image):
	""" Returns the power spectrum function of an image"""
	image = image.astype(float)
	ps_img = abs(pow(fft2(image), 2))
	return ps_img

def normalise(image):
	return image/np.amax(image)

#--------------
#	Making measurements
#--------------

def make_measurements(model, vis):
	''' specify which vis we are taking measurements from: H,V or A, B'''

	f = interpolate.interp2d(model.U_vals, model.V_vals, vis, kind='cubic')
	sampledVis = []
	for i in range(len(model.uMeters)):
		sampledVis.append(f(model.uMeters[i], model.vMeters[i])[0])
		
	sampledVis = np.array(sampledVis)	
	#--

	return sampledVis

#--------------
#	MCFOST
#--------------

def write_para(model):
	''' takes in a model with set parameters, and modified a parent parameter file with the new parameters'''

	in_f = open(model.parentPara, 'rU')
	out_f = open(model.modelPara, 'w')

	for i in range(73):
		line = in_f.readline()
		line = line.split()

		if i == 18:
			if model.grid:
				line[0] = str(model.grid)
				line[1] = str(model.grid)
			if model.size:
				line[2] = str(model.size)

		if i == 21:
			line[0] = str(model.distance)
		
		if i == 29:
			line[0] = str(model.imageSym)
		if i == 30:
			line[0] = str(model.centralSym)
		if i == 31:
			line[0] = str(model.axialSym)

		if i == 45:
			line[0] = str(model.dust_mass)
			
		if i == 47:
			line[0] = str(model.rin)

		if i == 70:
			if model.radius:
				line[1] = str(model.radius)
			if model.mass:
				line[2] = str(model.mass)
			if model.x:
				line[3] = str(model.x)
			if model.y:
				line[4] = str(model.y)
			if model.z:
				line[5] = str(model.z)

		out_f.write(' '.join(line) + '\n')

	in_f.close()
	out_f.close()

def make_model(model, disp_mcfost = False):
	''' return the fits file for a given wavelength and parameter file'''
	wavelength = str(model.wavelength)
	para_file = model.modelPara
	#  stdout = subprocess.PIPE
	start = time.time()
	if disp_mcfost == False:
		subprocess.call(['mcfost', para_file], stdout = subprocess.PIPE)
		subprocess.call(['mcfost', para_file, '-img', wavelength, '-rt'], stdout = subprocess.PIPE)
	else:
		subprocess.call(['mcfost', para_file])
		subprocess.call(['mcfost', para_file, '-img', wavelength, '-rt'])

	end = time.time()
	model_time = end - start
	try:
		f = open('data_' + wavelength + '/RT.fits.gz')
		f.close()
		subprocess.call(['mv', 'data_' + wavelength + '/RT.fits.gz', 'RT.fits.gz'])
	except OSError as e:
		print('File not made')
		sys.exit()
	
	subprocess.call(['gunzip', '-f', 'RT.fits.gz'])
	data = fits.getdata('RT.fits')
	return data, model_time

def delete_old(wavelength, mcfostPath):
	''' MCFOST doesn't overwrite old files - so you have to delete them yourself '''
	subprocess.call("rm -r data_*", shell = True)

#--------------
#	PLOTTING
#--------------
def correlation(epoch, individual, visHV, visAB):

	maxBL = max(individual.blengths)
	bl = individual.blengths
	blCols = bl
	blCols = blCols[bl <= maxBL]
	bl = bl[bl <= maxBL]

	title = 'Correlation'

	fig = plt.figure(figsize = (8,5))
	ax = fig.add_subplot(111)

	data_vis = individual.obsData.vhvvu
	data_err = individual.obsData.vhvvuerr
	model_vis = visAB
	chi2errAB = np.sum(pow((data_vis - model_vis)/data_err,2))
	scatPlt = ax.scatter(model_vis, data_vis, s = 10, marker = 'x',  edgecolors = 'none', c = individual.blengths, label = 'model')
	a, b, c =ax.errorbar(model_vis, data_vis, yerr = data_err, marker = '', 
		ls = '', alpha = 0.5, capsize = 0)
	barColor = scatPlt.to_rgba(blCols)
	c[0].set_color(barColor)

	data_vis = individual.obsData.vhvv
	data_err = individual.obsData.vhvverr
	model_vis = visHV
	chi2errHV = np.sum(pow((data_vis - model_vis)/data_err,2))
	scatPlt = ax.scatter(model_vis, data_vis, s = 10, marker = 'x',  edgecolors = 'none', c = individual.blengths, label = 'model')
	a, b, c =ax.errorbar(model_vis, data_vis, yerr = data_err, marker = '', 
		ls = '', alpha = 0.5, capsize = 0)
	barColor = scatPlt.to_rgba(blCols)
	c[0].set_color(barColor)

	ax.set_ylim([min(data_vis) - 0.01, max(data_vis) + 0.01])
	ax.set_xlim([min(data_vis) - 0.01, max(data_vis) + 0.01])

	chi2redHV = chi2errHV/(individual.n - individual.paraN)
	chi2redAB = chi2errAB/(individual.n - individual.paraN)
	chi2red = 0.5*(chi2redHV + chi2redAB)
	chi = '$\chi^2$ error: ' + str(round(chi2red, 2))
	plt.title(title)
	plt.plot(np.linspace(min(data_vis) - 0.1,max(data_vis) + 0.1,10), np.linspace(min(data_vis) - 0.1,max(data_vis) + 0.1,10))
	plt.colorbar(scatPlt, label='Baseline length (m)')
	plt.xlabel('Model Visibility Ratio')
	plt.ylabel('Data Visibility Ratio')
	plt.text(min(data_vis), max(data_vis), chi)
	plt.savefig(individual.resFolder + "/correlation/c" + str(epoch) + ".png")
	# plt.savefig("X_results/correlation/" + individual.resFolder + str(z) + ".png")
	plt.close()

def plot_vis2d(epoch, individual, sampledVis, stokes = None):

	# default is to show stokes Q
	if stokes == 'U':
		data_vis = individual.obsData.vhvvu
		data_err = individual.obsData.vhvvuerr
		title = 'Sampled Visibility Ratio (Stokes U)'
		chi2err = np.sum(pow((individual.obsData.vhvvu - sampledVis)/individual.obsData.vhvvuerr,2))
	else:
		data_vis = individual.obsData.vhvv
		data_err = individual.obsData.vhvverr
		title = 'Sampled Visibility Ratio (Stokes Q)'
		chi2err = np.sum(pow((individual.obsData.vhvv - sampledVis)/individual.obsData.vhvverr,2))
		
	chi2red = chi2err/(individual.n - individual.paraN)
	chi = '$\chi^2$ error: ' + str(round(chi2red, 2))
	model_vis = sampledVis

	maxBL = max(individual.blengths)
	bl = individual.blengths
	blCols = bl
	blCols = blCols[bl <= maxBL]
	bl = bl[bl <= maxBL]

	fig = plt.figure(figsize = (8,5))
	ax = fig.add_subplot(111)
	scatPlt = ax.scatter(individual.angles, model_vis, s = 100, marker = '.',  edgecolors = 'none', c = individual.blengths, label = 'model')
	ax.scatter(individual.angles, data_vis, marker = 'x', edgecolors = 'none', c = individual.blengths, label = 'data')
	a, b, c =ax.errorbar(individual.angles, data_vis, yerr = data_err, marker = '', 
		ls = '', alpha = 0.5, capsize = 0)
	barColor = scatPlt.to_rgba(blCols)
	c[0].set_color(barColor)
	ax.set_ylim([min(data_vis) - 0.01, max(data_vis) + 0.01])
	plt.title(title)
	plt.colorbar(scatPlt, label='Baseline length (m)')
	plt.xlabel('Baseline Azimuth Angle (radians)')
	plt.ylabel('Polarised Visibility Ratio')
	plt.legend()
	plt.text(-1.8, min(data_vis) + (max(data_vis) - min(data_vis))/20, chi)
	plt.savefig(individual.resFolder + "/" + stokes + "_animate/" + stokes + str(epoch) + ".png")
	plt.close()
	plt.show()

	fig = plt.figure(figsize = (6,4.5))
	ax = fig.add_subplot(211)
	scatPlt = ax.scatter(individual.angles, model_vis, marker = 'x', c = individual.blengths, label = 'model')
	ax.set_ylim([min(data_vis) - 0.01, max(data_vis) + 0.01])
	plt.colorbar(scatPlt, label='Baseline length (m)')
	plt.title('Model (Stokes' + stokes + ')')
	plt.xlabel('Baseline Azimuth Angle (radians)')
	plt.ylabel('Polarised Visibility Ratio')
	plt.text(-1.8, min(data_vis) + (max(data_vis) - min(data_vis))/20, chi)
	plt.close()

	ax = fig.add_subplot(212)
	ax.scatter(individual.angles, data_vis, marker = '+', c = individual.blengths, label = 'data')
	a, b, c =ax.errorbar(individual.angles, data_vis, yerr = data_err, marker = '', 
		ls = '', alpha = 0.5, capsize = 0 )
	barColor = scatPlt.to_rgba(blCols)
	c[0].set_color(barColor)
	ax.set_ylim([min(data_vis) - 0.01, max(data_vis) + 0.01])
	plt.colorbar(scatPlt, label='Baseline length (m)')
	plt.title('Data (Stokes' + stokes + ')')
	plt.xlabel('Baseline Azimuth Angle (radians)')
	plt.ylabel('Polarised Visibility Ratio')
	plt.close()

