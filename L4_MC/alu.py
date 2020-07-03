import os
import sys
import numpy
import neuron
import pickle
import requests
import datetime
import peakutils
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from bs4 import BeautifulSoup
from pyitlib import discrete_random_variable as drv
from L1_DAC import run as l1_dac
from L23_DBC import run as l23_dbc
from L4_NGC import run as l4_ngc
from L5_BP import run as l5_bp
from L6_SBC import run as l6_sbc
from L1_HAC import run as l1_hac
from L23_MC import run as l23_mc
from L4_DBC import run as l4_dbc
from L5_SBC import run as l5_sbc
from L6_BP import run as l6_bp
from L1_SAC import run as l1_sac
from L23_BTC import run as l23_btc
from L4_BP import run as l4_bp
from L5_NGC import run as l5_ngc
from L6_MC import run as l6_mc
from L1_NGCDA import run as l1_ngcda
from L23_NBC import run as l23_nbc
from L4_MC import run as l4_mc
from L5_DBC import run as l5_dbc
from L6_NGC import run as l6_ngc
from L23_BP import run as l23_bp
from L4_SBC import run as l4_sbc
from L6_DBC import run as l6_dbc
from L23_LBC import run as l23_lbc
from L6_LBC import run as l6_lbc

plt.style.use('seaborn-deep')
plt.rcParams.update({'font.size': 12})

root_path = os.getcwd()
results_dir = './Results/'

cells = []
models = []
syn_ = []
nclist = []

syns = []
stim = []
ncstim = []
recordings = {}
timeRecordings = {}

message = []

n_neurons = 1
weight = .04 # uS
simTime = 1000
numSlots = 1000
numBins = 200
slotSize = int(simTime / numSlots)
binSize = int(simTime / numBins)
slotsPerBin = int(binSize / slotSize)
spiking_frequency = float(sys.argv[1])

layers = ['L1', 'L23', 'L4', 'L5', 'L6']
cp_df = pd.read_pickle('connection_probability.pkl')
pc_df = pd.read_pickle('peak_conductances.pkl')
sc_df = pd.read_pickle('synapses_per_connection.pkl')

class xor(object):
	models = ['L1_DAC', 'L1_HAC', 'L23_DBC', 'L23_MC', 'L4_DBC']
	cells = []
	thres = 5
	weight = .04

	def __init__(self):
		self.create_cells()
		self.build_gate()
		self.get_terminals()

	def create_cells(self):
		for i in range(len(self.models)):
			os.chdir(self.models[i])
			self.cells.append(eval(self.models[i].lower()).create_cell(add_synapses=False))
			os.chdir(root_path)
			
	def build_gate(self):
		establishSynapses(self.cells[0], self.cells[2], self.thres, self.weight)
		establishSynapses(self.cells[1], self.cells[3], self.thres, self.weight)
		establishSynapses(self.cells[0], self.cells[3], self.thres, -self.weight)
		establishSynapses(self.cells[1], self.cells[2], self.thres, -self.weight)
		establishSynapses(self.cells[2], self.cells[4], self.thres, self.weight)
		establishSynapses(self.cells[3], self.cells[4], self.thres, self.weight)

	def get_terminals(self):
		self.input1 = self.cells[0].soma[0]
		self.input2 = self.cells[1].soma[0]
		self.output = self.cells[4].axon[0]

class and1(object):
	models = ['L23_MC', 'L23_NBC', 'L1_HAC']
	cells = []
	thres = -64

	def __init__(self):
		self.create_cells()
		self.build_gate()
		self.get_terminals()

	def create_cells(self):
		for i in range(len(self.models)):
			os.chdir(self.models[i])
			self.cells.append(eval(self.models[i].lower()).create_cell(add_synapses=False))
			os.chdir(root_path)
	
	def build_gate(self):
		establishSynapses(self.cells[0], self.cells[2], self.thres)
		establishSynapses(self.cells[1], self.cells[2], self.thres)

	def get_terminals(self):
		self.input1 = self.cells[0].soma[0]
		self.input2 = self.cells[1].soma[0]
		self.output = self.cells[2].soma[0]

class halfAdder(object):

	models = ['L23_NBC', 'L23_NBC']
	ext_cells = []

	def __init__(self):
		self.instantiate_gates()
		self.create_cells()
		self.get_cells()
		self.build_circuit()
		self.get_terminals()

	def create_cells(self):
		for i in range(len(self.models)):
			os.chdir(self.models[i])
			self.ext_cells.append(eval(self.models[i].lower()).create_cell(add_synapses=False))
			os.chdir(root_path)

	def instantiate_gates(self):
		self.xorGate = xor()
		self.andGate = and1()
	
	def get_cells(self):
		self.cells = [self.xorGate.cells[0], self.xorGate.cells[1], self.andGate.cells[0], self.andGate.cells[1], self.xorGate.cells[4], self.andGate.cells[2]]

	def build_circuit(self):
		connectGates(self.ext_cells[0].soma[0], self.xorGate.input1, numConnections=5)
		connectGates(self.ext_cells[0].soma[0], self.andGate.input1, numConnections=5)
		connectGates(self.ext_cells[1].soma[0], self.xorGate.input2, numConnections=5)
		connectGates(self.ext_cells[1].soma[0], self.andGate.input2, numConnections=5)

	def get_terminals(self):
		self.inA = self.ext_cells[0].dend[0]
		self.inB = self.ext_cells[1].dend[0]
		self.in_1a = self.xorGate.input1
		self.in_1b = self.xorGate.input2
		self.in_2a = self.andGate.input1
		self.in_2b = self.andGate.input2
		self.mout = self.xorGate.output
		self.rout = self.andGate.output

class halfSubtracter(object):
	def __init__(self):
		self.instantiate_gates()
		self.get_cells()
		self.build_circuit()
		self.get_terminals()

	def instantiate_gates(self):
		self.xorGate1 = xor()
		self.xorGate2 = xor()
	
	def get_cells(self):
		self.cells = [self.xorGate1.cells[0], self.xorGate1.cells[1], self.xorGate2.cells[0], self.xorGate2.cells[1], self.xorGate1.cells[4], self.xorGate2.cells[4]]

	def build_circuit(self):
		pass

	def get_terminals(self):
		self.in_1a = self.xorGate1.input1
		self.in_1b = self.xorGate1.input2
		self.in_2a = self.xorGate2.input1
		self.in_2b = self.xorGate2.input2
		self.mout = self.xorGate1.output
		self.rout = self.xorGate2.output

def getAllModels():
	""" Get all imported neuron models. """
	global models
	models = [key for key in sys.modules.keys() if '.run' not in key and 'L' == key[0]]

def printTimeStamp(fname, fmt='%Y%m%d%H%M%S_{fname}'):
	return datetime.datetime.now().strftime(fmt).format(fname=fname)

def loadSimulationParameters():
	""" Load parameters for simulation. """
	os.chdir(models[0])
	eval(models[0].lower()).init_simulation()
	os.chdir(root_path)

def getConnectionProbability(source_cell, target_cell):
	""" Get connection probability from local file. """
	return cp_df[source_cell][target_cell]

def connectGates(preSyn, postSyn, numConnections=1, thres=10, w=.04):
	for i in range(numConnections):
		# for j in range(synapsesPerConnection):
		syn = neuron.h.ExpSyn(0.5, sec = postSyn) # Change to j when uncomment inner for loop.
		syns.append(syn)
		# print(syn.e)
		nc = neuron.h.NetCon(preSyn(0.5)._ref_v, syn, sec=preSyn)
		# Excitatory or inhibitory connections can be defined by the FNCCH technique
		# Implementation details can be found at https://doi.org/10.1371/journal.pcbi.1006381
		# nc.weight[0] = weight if fncch[cells.index(preSyn)][cells.index(postSyn)] > 0 else -weight
		
		nc.weight[0] = w # getPeakConductance(sourceModel, targetModel) / 1000 # Divided by 100 to get a similar value to the tutorial or by 1000 to get uS values.
		nc.delay = 0
		nc.threshold = thres
		# spikeTimes = neuron.h.Vector()
		# nc.record(spikeTimes)
		nclist.append(nc)
		# spikeTimes.append(spikeTimes)

def establishSynapses(preSyn, postSyn, thres=10, w=.04):
	"""Establish synaptic connections routine.

	Establish synaptic connections between source_cell and
	target_cell according to connection probability.

	Parameters
	----------
	src : tuple
		The index of the source cell.
	tgt : tuple
		The index of the target cell.

	Returns
	-------
	n/a
	"""
	# print('Establishing synapses.')
	# for preSyn in cells:
		# for postSyn in cells:
	# Extract the name of the cell model.
	sourceIndex = (str(preSyn).find('_') + 1, str(preSyn).rfind('_'))
	targetIndex = (str(postSyn).find('_') + 1, str(postSyn).rfind('_'))
	sourceModel = str(preSyn)[sourceIndex[0]:sourceIndex[1]]
	targetModel = str(postSyn)[targetIndex[0]:targetIndex[1]]

	# Check inconsistencies in L1_SAC/L1_SLAC.
	if sourceModel == 'L1_SLAC':
		sourceModel = 'L1_SAC'
	if targetModel == 'L1_SLAC':
		targetModel = 'L1_SAC'

	# Uncomment line below if using FNCCH and indent block of code underneath it.
	# if fncch[cells.index(preSyn)][cells.index(postSyn)] != 0:

	# Calculate the number of connections between preSyn and postSyn.
	numConnections = int(round((getConnectionProbability(sourceModel, targetModel) / 100) * len(postSyn.dend), 0))
	# synapsesPerConnection = int(getSynapsesPerConnection(sourceModel, targetModel))

	for i in range(numConnections):
		# for j in range(synapsesPerConnection):
		syn = neuron.h.ExpSyn(0.5, sec = postSyn.dend[i]) # Change to j when uncomment inner for loop.
		syns.append(syn)
		nc = neuron.h.NetCon(preSyn.soma[0](0.5)._ref_v, syn, sec=preSyn.soma[0])
		# Excitatory or inhibitory connections can be defined by the FNCCH technique
		# Implementation details can be found at https://doi.org/10.1371/journal.pcbi.1006381
		# nc.weight[0] = weight if fncch[cells.index(preSyn)][cells.index(postSyn)] > 0 else -weight
		nc.weight[0] = w # getPeakConductance(sourceModel, targetModel) / 1000 # Divided by 100 to get a similar value to the tutorial or by 1000 to get uS values.
		nc.delay = 0
		nc.threshold = thres
		# spikeTimes = neuron.h.Vector()
		# nc.record(spikeTimes)
		nclist.append(nc)
		# spikeTimes.append(spikeTimes)

def getRasterPlot(spikeTimes, fileFormat='pdf'):
	print('Plotting raster plot.')
	plt.eventplot(spikeTimes)
	plt.ylabel('Cell ID')
	plt.xlabel('Time')
	plt.xlim((0, simTime))
	plt.grid(which='both', axis='x', linestyle=':')
	# plt.title('14 gates')
	plt.tight_layout()
	plt.savefig('{}{}.{}'.format(results_dir, printTimeStamp('raster_plot'), fileFormat), format=fileFormat, dpi=150)
	plt.close()

def plotOutput(recordings, fileFormat='pdf'):
	print('Plotting spiking activity.')
	for key in list(recordings.keys()):
		plt.plot(numpy.linspace(0, simTime, len(recordings[key])), recordings[key], label='Soma')
		plt.xlabel('Time (ms)')
		plt.ylabel('Membrane Potential (mV)')
		plt.grid(which='both', axis='both')
		plt.ylim(-80, 20)
		plt.margins(0, 0)
		plt.legend()
		plt.title('{}'.format(key))
		plt.tight_layout()
		plt.savefig('{}{}.{}'.format(results_dir, printTimeStamp(key), fileFormat), format=fileFormat, dpi=150)
		plt.close()

def accuracy(X, Y):
	probs = numpy.zeros((2, 2))
	for c1 in set(X):
		for c2 in set(Y):
				probs[c1][c2] = numpy.mean(numpy.logical_and(X == c1, Y == c2))

	return numpy.sum([probs[0][0], probs[1][1]]) / numpy.sum(probs)

def bitwise_half_adder(X, Y):
	c = numpy.bitwise_and(X, Y)
	s = numpy.bitwise_xor(X, Y)
	return s, c

def bitwise_half_subtracter(X, Y):
	d = numpy.bitwise_xor(X, Y)
	bout = numpy.zeros(len(X))
	for i in range(len(X)):
		if X[i] == 0 and Y[i] == 1:
			bout[i] = 1
	return d, bout

def getSpikesAsBits():
	global spiking_frequency
	slotSize = 1
	# Creates array to hold the bit train.
	spikesAsBits = numpy.zeros((len(recordings.keys()), numSlots), dtype=numpy.int8)
	# Create list to hold the time of the spikes.
	spikeTimes = []
	# Finds the numeric index of the spikes.
	for key, i in zip(list(recordings.keys()), range(len(recordings.keys()))):
		# 0.025 (see constants.hoc)
		spikesIndexes = peakutils.indexes(recordings[key], thres=0, min_dist=1, thres_abs=True) * 0.025
		# Populates the list with spike times.
		spikeTimes.append(list(spikesIndexes))
		# Populates the bit train array.
		for j in spikesIndexes:
			spikesAsBits[i][int(round(j / slotSize))] = 1

	getRasterPlot(spikeTimes)

	primaryRes, secondaryRes = bitwise_half_adder(spikesAsBits[0], spikesAsBits[1])
	# primaryRes, secondaryRes = bitwise_half_subtracter(spikesAsBits[0], spikesAsBits[1])

	# Change results for accuracy
	accOp = numpy.mean(numpy.array([accuracy(primaryRes, spikesAsBits[-2]), accuracy(secondaryRes, spikesAsBits[-1])]))

	with open('checking_spiking_train.txt', 'a+') as text:
		text.write('1-3: {}, 2-4: {}\n'.format(accuracy(spikesAsBits[0], spikesAsBits[2]), accuracy(spikesAsBits[1], spikesAsBits[3])))
	text.close()

	with open('{}halfAdder.txt'.format(results_dir), 'a+') as f:
			f.write('({}, {}), '.format(spiking_frequency, accOp))
	f.close()

def runSimulation():
	global spiking_frequency

	circuit = halfAdder()

	stim = []
	syn_ = []
	ncstim = []

	stim2 = []
	syn_2 = []
	ncstim2 = []

	stim3 = []
	syn_3 = []
	ncstim3 = []

	stim4 = []
	syn_4 = []
	ncstim4 = []

	seedA = numpy.random.randint(100, size=1)[0]
	seedB = numpy.random.randint(100, size=1)[0]

	stim = neuron.h.NetStim()
	syn_ = neuron.h.ExpSyn(0.5, sec=circuit.inA)
	stim.start = 0
	stim.number = 1e9
	stim.noise = 1
	stim.seed(seedA)
	stim.interval = 1000 / spiking_frequency
	ncstim = neuron.h.NetCon(stim, syn_)
	ncstim.delay = 0
	ncstim.weight[0] = weight
	syn_.tau = 2

	stim2 = neuron.h.NetStim()
	syn_2 = neuron.h.ExpSyn(0.5, sec=circuit.inB)
	stim2.start = 0
	stim2.number = 1e9
	stim2.noise = 1
	stim2.seed(seedB)
	stim2.interval = 1000 / spiking_frequency
	ncstim2 = neuron.h.NetCon(stim2, syn_2)
	ncstim2.delay = 0
	ncstim2.weight[0] = weight
	syn_2.tau = 2

	# stim3 = neuron.h.NetStim()
	# syn_3 = neuron.h.ExpSyn(0.5, sec=circuit.in_2a)
	# stim3.start = 0
	# stim3.number = 1e9
	# stim3.noise = 1
	# stim3.seed(seed1)
	# stim3.interval = 1000 / spiking_frequency
	# ncstim3 = neuron.h.NetCon(stim3, syn_3)
	# ncstim3.delay = 0
	# ncstim3.weight[0] = weight
	# syn_3.tau = 2

	# stim4 = neuron.h.NetStim()
	# syn_4 = neuron.h.ExpSyn(0.5, sec=circuit.in_2b)
	# stim4.start = 0
	# stim4.number = 1e9
	# stim4.noise = 1
	# stim4.seed(seed2)
	# stim4.interval = 1000 / spiking_frequency
	# ncstim4 = neuron.h.NetCon(stim4, syn_4)
	# ncstim4.delay = 0
	# ncstim4.weight[0] = weight
	# syn_4.tau = 2

	for cell, idNum in zip(circuit.cells, range(len(circuit.cells))):
		recordings['{}_{}'.format(cell, idNum)] = neuron.h.Vector()
		timeRecordings['{}_{}'.format(cell, idNum)] = neuron.h.Vector()
		recordings['{}_{}'.format(cell, idNum)].record(cell.soma[0](0.5)._ref_v)
		timeRecordings['{}_{}'.format(cell, idNum)].record(neuron.h._ref_t)

	print('The simulation has just started.')
	neuron.h.tstop = simTime
	neuron.h.run()

def exportData():
	print('Exporting recorded data.')
	# pickle.dump(recordings, open(os.getcwd() + '/Results/' + printTimeStamp('recordings.pkl'), 'wb'))
	print('Nothing was exported.')

getAllModels()
loadSimulationParameters()
runSimulation()
exportData()
# plotOutput(recordings)
getSpikesAsBits()