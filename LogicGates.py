import os
import sys
import numpy
import neuron
import pickle
import requests
import datetime
import peakutils
import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
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
from scipy.interpolate import griddata

plt.style.use('seaborn-deep')
plt.rcParams.update({'font.size': 12})

root_path = os.getcwd()
resultsFolder = './Results/'
cells = []
models = []
syn_ = []
nclist = []
# spikeTimes = []
syns = []
stim = []
ncstim = []
message = []
recordings = {}
timeRecordings = {}
dendRecordings = {}
synCurrentRec = {}
eLeak = {}
andGate = [] # Variable to hold the AND gates
# andGate2 = None

n_neurons = 1
weight = float(sys.argv[2]) # uS
simTime = 1000
numSlots = 1000
numBins = 50
pMsg = 0.5
slotSize = int(simTime / numSlots)
binSize = int(simTime / numBins)
slotsPerBin = int(binSize / slotSize)
freq = float(sys.argv[1])

layers = ['L1', 'L23', 'L4', 'L5', 'L6']
cp_df = pd.read_pickle('connection_probability.pkl')
pc_df = pd.read_pickle('peak_conductances.pkl')
sc_df = pd.read_pickle('synapses_per_connection.pkl')
# fncch = numpy.load('network_connections.npy')

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

class or1(object):
	models = ['L1_DAC', 'L1_SAC', 'L23_LBC']
	cells = []
	thres = 5

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
		self.output = self.cells[2].axon[0]

class or2(object):
	models = ['L23_MC', 'L23_NBC', 'L1_DAC']
	cells = []
	thres = 5

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
		self.output = self.cells[2].axon[0]

class or3(object):
	models = ['L4_DBC', 'L23_BTC', 'L5_BP']
	cells = []
	thres = 5

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
		self.output = self.cells[2].axon[0]

class or4(object):
	models = ['L4_DBC', 'L23_DBC', 'L5_BP']
	cells = []
	thres = 5

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
		self.output = self.cells[2].axon[0]

class or5(object):
	models = ['L1_DAC', 'L1_HAC', 'L23_MC']
	cells = []
	thres = 5

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
		self.output = self.cells[2].axon[0]
	

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
	
class dynamicGate(object):
	models = ['L23_MC', 'L23_NBC', 'L1_DAC']
	cells = []
	thres = 10

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
		establishSynapses(self.cells[0], self.cells[2], thres=self.thres, w=.03) # .03 for AND .06 for OR
		establishSynapses(self.cells[1], self.cells[2], thres=self.thres, w=.03)

	def get_terminals(self):
		self.input1 = self.cells[0].soma[0]
		self.input2 = self.cells[1].soma[0]
		self.output = self.cells[2].soma[0]

class and2(object):
	models = ['L5_SBC', 'L23_MC', 'L4_SBC']
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
		self.output = self.cells[2].axon[0]

class and3(object):
	models = ['L6_MC', 'L4_MC', 'L1_DAC']
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
		self.output = self.cells[2].axon[0]

class randomGate1(object):
	models = ['L6_LBC', 'L4_BP', 'L1_NGCDA']
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
		self.output = self.cells[2].axon[0]
	
class randomGate2(object):
	models = ['L1_SAC', 'L5_BP', 'L1_DAC']
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
		self.output = self.cells[2].axon[0]

class randomGate3(object):
	models = ['L4_SBC', 'L6_MC', 'L23_DBC']
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
		self.output = self.cells[2].axon[0]

class randomCircuit(object):
	def __init__(self):
		self.instantiate_gates()
		self.get_cells()
		self.build_circuit()
		self.get_terminals()

	def instantiate_gates(self):
		self.orGate1 = randomGate1()
		self.orGate2 = randomGate2()
		self.andGate = randomGate3()
	
	def get_cells(self):
		self.cells = [self.orGate1.cells[0], self.orGate1.cells[1], self.orGate2.cells[0], self.orGate2.cells[1], self.andGate.cells[2]]

	def build_circuit(self):
		connectGates(self.orGate1.output, self.andGate.input1)
		connectGates(self.orGate2.output, self.andGate.input2)

	def get_terminals(self):
		self.in1 = self.orGate1.input1
		self.in2 = self.orGate1.input2
		self.in3 = self.orGate2.input1
		self.in4 = self.orGate2.input2
		self.out = self.andGate.output

class circuitA(object):
	'''
	= OR\
		 =AND-
	= OR/
	'''
	def __init__(self):
		self.instantiate_gates()
		self.get_cells()
		self.build_circuit()
		self.get_terminals()

	def instantiate_gates(self):
		self.orGate1 = or2()
		self.orGate2 = or2()
		self.andGate = dynamicGate() #and1()
	
	def get_cells(self):
		self.cells = [self.orGate1.cells[0], self.orGate1.cells[1], self.orGate2.cells[0], self.orGate2.cells[1], self.andGate.cells[2]]

	def build_circuit(self):
		connectGates(self.orGate1.output, self.andGate.input1)
		connectGates(self.orGate2.output, self.andGate.input2)

	def get_terminals(self):
		self.in1 = self.orGate1.input1
		self.in2 = self.orGate1.input2
		self.in3 = self.orGate2.input1
		self.in4 = self.orGate2.input2
		self.out = self.andGate.output

class circuitB(object):
	'''
	= OR \
		  =AND-
	= AND/
	'''
	def __init__(self):
		self.instantiate_gates()
		self.get_cells()
		self.build_circuit()
		self.get_terminals()

	def instantiate_gates(self):
		self.orGate1 = or1() #or2()
		self.orGate2 = and2() #and1()
		self.andGate = and2() #and1()
	
	def get_cells(self):
		self.cells = [self.orGate1.cells[0], self.orGate1.cells[1], self.orGate2.cells[0], self.orGate2.cells[1], self.andGate.cells[2]]

	def build_circuit(self):
		connectGates(self.orGate1.output, self.andGate.input1)
		connectGates(self.orGate2.output, self.andGate.input2)

	def get_terminals(self):
		self.in1 = self.orGate1.input1
		self.in2 = self.orGate1.input2
		self.in3 = self.orGate2.input1
		self.in4 = self.orGate2.input2
		self.out = self.andGate.output

class circuitC(object):
	'''
	= AND\
		  =OR-
	= AND/
	'''
	def __init__(self):
		self.instantiate_gates()
		self.get_cells()
		self.build_circuit()
		self.get_terminals()

	def instantiate_gates(self):
		self.orGate1 = and1()
		self.orGate2 = and1()
		self.andGate = or2()
	
	def get_cells(self):
		self.cells = [self.orGate1.cells[0], self.orGate1.cells[1], self.orGate2.cells[0], self.orGate2.cells[1], self.andGate.cells[2]]

	def build_circuit(self):
		connectGates(self.orGate1.output, self.andGate.input1)
		connectGates(self.orGate2.output, self.andGate.input2)

	def get_terminals(self):
		self.in1 = self.orGate1.input1
		self.in2 = self.orGate1.input2
		self.in3 = self.orGate2.input1
		self.in4 = self.orGate2.input2
		self.out = self.andGate.output

class circuitC_inh(object):
	'''
	= AND\
		  =OR-
	= AND/
	'''
	def __init__(self):
		self.instantiate_gates()
		self.get_cells()
		self.build_circuit()
		self.get_terminals()

	def instantiate_gates(self):
		self.orGate1 = and1()
		self.orGate2 = and1()
		self.andGate = or2()
	
	def get_cells(self):
		self.cells = [self.orGate1.cells[0], self.orGate1.cells[1], self.orGate2.cells[0], self.orGate2.cells[1], self.andGate.cells[2]]

	def build_circuit(self):
		connectGates(self.orGate1.output, self.andGate.input1)
		connectGates(self.orGate2.output, self.andGate.input2, w=-.04)

	def get_terminals(self):
		self.in1 = self.orGate1.input1
		self.in2 = self.orGate1.input2
		self.in3 = self.orGate2.input1
		self.in4 = self.orGate2.input2
		self.out = self.andGate.output

class circuitD(object):
	'''
	= AND\
		  =AND-
	= AND/
	'''
	def __init__(self):
		self.instantiate_gates()
		self.get_cells()
		self.build_circuit()
		self.get_terminals()

	def instantiate_gates(self):
		self.orGate1 = and1()
		self.orGate2 = and1()
		self.andGate = or2() # Change synaptic weights (CIRCUIT C)
		# Cascade gates
		self.andGate1 = dynamicGate()
		# self.andGate2 = dynamicGate()
	
	def get_cells(self):
		self.cells = [self.orGate1.cells[0], self.orGate1.cells[1], self.orGate2.cells[0], self.orGate2.cells[1], self.andGate.cells[2], self.andGate1.cells[0], self.andGate1.cells[1], self.andGate1.cells[2]]#, self.andGate2.cells[0], self.andGate2.cells[1], self.andGate2.cells[2]]

	def build_circuit(self):
		connectGates(self.orGate1.output, self.andGate.input1)
		connectGates(self.orGate2.output, self.andGate.input2)
		# Cascade gates
		connectGates(self.andGate.output, self.andGate1.input1)
		# connectGates(self.andGate1.output, self.andGate2.input1)

	def get_terminals(self):
		self.in1 = self.orGate1.input1
		self.in2 = self.orGate1.input2
		self.in3 = self.orGate2.input1
		self.in4 = self.orGate2.input2
		self.out = self.andGate.output
		# Cascade gates
		self.in5 = self.andGate1.input2
		# self.in6 = self.andGate2.input2
		self.out3 = self.andGate1.output

class circuitE(object):
	'''
	= OR \
		  =OR-
	= AND/
	'''
	def __init__(self):
		self.instantiate_gates()
		self.get_cells()
		self.build_circuit()
		self.get_terminals()

	def instantiate_gates(self):
		self.orGate1 = or2()
		self.orGate2 = and1()
		self.andGate = or2()
	
	def get_cells(self):
		self.cells = [self.orGate1.cells[0], self.orGate1.cells[1], self.orGate2.cells[0], self.orGate2.cells[1], self.andGate.cells[2]]

	def build_circuit(self):
		connectGates(self.orGate1.output, self.andGate.input1)
		connectGates(self.orGate2.output, self.andGate.input2)

	def get_terminals(self):
		self.in1 = self.orGate1.input1
		self.in2 = self.orGate1.input2
		self.in3 = self.orGate2.input1
		self.in4 = self.orGate2.input2
		self.out = self.andGate.output

class circuitF(object):
	'''
	= OR\
		  =OR-
	= OR/
	'''
	def __init__(self):
		self.instantiate_gates()
		self.get_cells()
		self.build_circuit()
		self.get_terminals()

	def instantiate_gates(self):
		self.orGate1 = or2()
		self.orGate2 = or2()
		self.andGate = or2()
	
	def get_cells(self):
		self.cells = [self.orGate1.cells[0], self.orGate1.cells[1], self.orGate2.cells[0], self.orGate2.cells[1], self.andGate.cells[2]]

	def build_circuit(self):
		connectGates(self.orGate1.output, self.andGate.input1)
		connectGates(self.orGate2.output, self.andGate.input2)

	def get_terminals(self):
		self.in1 = self.orGate1.input1
		self.in2 = self.orGate1.input2
		self.in3 = self.orGate2.input1
		self.in4 = self.orGate2.input2
		self.out = self.andGate.output
		
def poissonFiring(n_cells, n_trials, p):
	"""
		Creates a poisson spiking data.
		Adapted from https://github.com/computational-neuroscience/Computational-Neuroscience-UW/blob/master/week-02/poisson_neuron.py
	"""

	spikes = numpy.random.rand(n_cells, n_trials)
	spikes[spikes < p] = 1
	spikes[spikes < 1] = 0

	return spikes

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
	numConnections = 3 # int(sys.argv[3])
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
	plt.savefig('{}{}.{}'.format(resultsFolder, printTimeStamp('raster_plot'), fileFormat), format=fileFormat, dpi=75)
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
		plt.savefig('{}{}.{}'.format(resultsFolder, printTimeStamp(key), fileFormat), format=fileFormat, dpi=75)
		plt.close()

	# for key in list(dendRecordings.keys()):
	# 	plt.plot(numpy.linspace(0, simTime, len(dendRecordings[key])), dendRecordings[key], label='Dend{}'.format(key[-1]))
	# plt.xlabel('Time (ms)')
	# plt.ylabel('Membrane Potential (mV)')
	# plt.grid(which='both', axis='both')
	# # plt.axvline(x=300, c='r')
	# # plt.axvline(x=550, c='r')
	# # plt.axvline(x=650, c='r')
	# plt.axhline(y=-20, c='g', linestyle=':')
	# plt.ylim(-80, 40)
	# plt.margins(0, 0)
	# # plt.legend()
	# plt.title('{}'.format(key))
	# plt.tight_layout()
	# plt.savefig('{}{}.{}'.format(resultsFolder, printTimeStamp(key), fileFormat), format=fileFormat, dpi=75)
	# plt.close()

	# for key in list(synCurrentRec.keys()):
	# 	plt.plot(numpy.linspace(0, simTime, len(synCurrentRec[key])), synCurrentRec[key], label='Synaptic Current')
	# 	plt.xlabel('Time (ms)')
	# 	plt.ylabel('Synaptic Current (nA)')
	# 	plt.grid(which='both', axis='both')
	# 	# plt.axvline(x=300, c='r')
	# 	# plt.axvline(x=550, c='r')
	# 	# plt.axvline(x=650, c='r')
	# 	# plt.axhline(y=0, c='g', linestyle=':')type(
	# 	# plt.ylim(-80, 40)
	# 	# plt.margins(0, 0)
	# 	plt.legend()
	# 	plt.title('{}'.format(key))
	# 	plt.tight_layout()
	# 	plt.savefig('{}{}_{}.{}'.format(resultsFolder, printTimeStamp(key), 'SynCur', fileFormat), format=fileFormat, dpi=75)
	# 	plt.close()

def accuracy(X, Y):
	probs = numpy.zeros((2, 2))
	for c1 in set(X):
		for c2 in set(Y):
				probs[c1][c2] = numpy.mean(numpy.logical_and(X == c1, Y == c2))

	return numpy.sum([probs[0][0], probs[1][1]]) / numpy.sum(probs)

def bitwise_circuitA(in1, in2, in3, in4):
	if len(set([in1.shape[0], in2.shape[0], in3.shape[0], in4.shape[0]])) == 1:
		realOutput = numpy.zeros(in1.shape[0], dtype=numpy.int8)
		for i in range(in1.shape[0]):
			if (in1[i] == 0 and in2[i] == 1 and in3[i] == 0 and in4[i] == 1) or (in1[i] == 0 and in2[i] == 1 and in3[i] == 1 and in4[i] == 0) or (in1[i] == 0 and in2[i] == 1 and in3[i] == 1 and in4[i] == 1) or (in1[i] == 1 and in2[i] == 0 and in3[i] == 0 and in4[i] == 1) or (in1[i] == 1 and in2[i] == 0 and in3[i] == 1 and in4[i] == 0) or (in1[i] == 1 and in2[i] == 0 and in3[i] == 1 and in4[i] == 1) or (in1[i] == 1 and in2[i] == 1 and in3[i] == 0 and in4[i] == 1) or (in1[i] == 1 and in2[i] == 1 and in3[i] == 1 and in4[i] == 0) or (in1[i] == 1 and in2[i] == 1 and in3[i] == 1 and in4[i] == 1):
				realOutput[i] = 1
			else:
				realOutput[i] = 0
		return realOutput
	else:
		raise ValueError('Inputs should have the same size!')

def bitwise_circuitB(in1, in2, in3, in4):
	if len(set([in1.shape[0], in2.shape[0], in3.shape[0], in4.shape[0]])) == 1:
		realOutput = numpy.zeros(in1.shape[0], dtype=numpy.int8)
		for i in range(in1.shape[0]):
			if (in1[i] == 0 and in2[i] == 1 and in3[i] == 1 and in4[i] == 1) or (in1[i] == 1 and in2[i] == 0 and in3[i] == 1 and in4[i] == 1) or (in1[i] == 1 and in2[i] == 1 and in3[i] == 1 and in4[i] == 1):
				realOutput[i] = 1
			else:
				realOutput[i] = 0
		return realOutput
	else:
		raise ValueError('Inputs should have the same size!')

def bitwise_circuitC(in1, in2, in3, in4):
	if len(set([in1.shape[0], in2.shape[0], in3.shape[0], in4.shape[0]])) == 1:
		realOutput = numpy.zeros(in1.shape[0], dtype=numpy.int8)
		for i in range(in1.shape[0]):
			if (in1[i] == 1 and in2[i] == 1) or (in3[i] == 1 and in4[i] == 1):
				realOutput[i] = 1
			else:
				realOutput[i] = 0
		return realOutput
	else:
		raise ValueError('Inputs should have the same size!')

def bitwise_circuitCplus2(in1, in2, in3, in4, in5, in6):
	if len(set([in1.shape[0], in2.shape[0], in3.shape[0], in4.shape[0]])) == 1:
		realOutput = numpy.zeros(in1.shape[0], dtype=numpy.int8)
		for i in range(in1.shape[0]):
			if (in1[i] == 1 and in2[i] == 1) or (in3[i] == 1 and in4[i] == 1) and in5[i] == 1 and in6[i] == 1:
				realOutput[i] = 1
			else:
				realOutput[i] = 0
		return realOutput
	else:
		raise ValueError('Inputs should have the same size!')

def bitwise_circuitCplus1(in1, in2, in3, in4, in5):
	if len(set([in1.shape[0], in2.shape[0], in3.shape[0], in4.shape[0]])) == 1:
		realOutput = numpy.zeros(in1.shape[0], dtype=numpy.int8)
		for i in range(in1.shape[0]):
			if (in1[i] == 1 and in2[i] == 1) or (in3[i] == 1 and in4[i] == 1) and in5[i] == 1:
				realOutput[i] = 1
			else:
				realOutput[i] = 0
		return realOutput
	else:
		raise ValueError('Inputs should have the same size!')

def bitwise_circuitD(in1, in2, in3, in4):
	if len(set([in1.shape[0], in2.shape[0], in3.shape[0], in4.shape[0]])) == 1:
		realOutput = numpy.zeros(in1.shape[0], dtype=numpy.int8)
		for i in range(in1.shape[0]):
			if (in1[i] == 1 and in2[i] == 1 and in3[i] == 1 and in4[i] == 1):
				realOutput[i] = 1
			else:
				realOutput[i] = 0
		return realOutput
	else:
		raise ValueError('Inputs should have the same size!')

def bitwise_circuitDplus1(in1, in2, in3, in4, in5):
	if len(set([in1.shape[0], in2.shape[0], in3.shape[0], in4.shape[0]])) == 1:
		realOutput = numpy.zeros(in1.shape[0], dtype=numpy.int8)
		for i in range(in1.shape[0]):
			if (in1[i] == 1 and in2[i] == 1 and in3[i] == 1 and in4[i] == 1 and in5[i] == 1):
				realOutput[i] = 1
			else:
				realOutput[i] = 0
		return realOutput
	else:
		raise ValueError('Inputs should have the same size!')

def bitwise_circuitDplus2(in1, in2, in3, in4, in5, in6):
	if len(set([in1.shape[0], in2.shape[0], in3.shape[0], in4.shape[0]])) == 1:
		realOutput = numpy.zeros(in1.shape[0], dtype=numpy.int8)
		for i in range(in1.shape[0]):
			if (in1[i] == 1 and in2[i] == 1 and in3[i] == 1 and in4[i] == 1 and in5[i] == 1 and in6[i] == 1):
				realOutput[i] = 1
			else:
				realOutput[i] = 0
		return realOutput
	else:
		raise ValueError('Inputs should have the same size!')

def getSpikesAsBits():
	global freq
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
			spikesAsBits[i][int(j / slotSize)] = 1
	
	acc = accuracy(bitwise_circuitCplus1(spikesAsBits[0], spikesAsBits[1], spikesAsBits[2], spikesAsBits[3], spikesAsBits[6]), spikesAsBits[-1])
	ratio = len(spikeTimes[-1]) / float(sys.argv[1])

	with open('./Results/circuitC_cascade_1.txt', 'a+') as f:
		f.write('[{}, {}, {}], '.format(freq, ratio, acc)) # ratio for circuit, accuracy for gates
	f.close()

	# with open('./Results/ratio_circuitD.txt', 'a+') as f:
	# 		f.write('({}, {}), '.format(freq, ratio))
	# f.close()
		
	getRasterPlot(spikeTimes)

# sys.argv[1]: frequency
# sys.argv[2]: weight
# sys.argv[3]: synapses
# sys.arvg[3]: freq_inhibitory

def createCells():
	""" Load cell models. """
	global cells
	if n_neurons <= 0:
		raise ValueError('Invalid number!')
	else:
		cells_per_layer = int(n_neurons / len(layers))
		remaining_cells = int(n_neurons % len(layers))
		cells = []
		for layer in layers:
			type_per_layer = [cell for cell in models if layer in cell]
			i = 0
			for j in range(cells_per_layer):
				i = i if i < len(type_per_layer) else 0
				os.chdir(type_per_layer[i])
				cells.append(eval(type_per_layer[i].lower()).create_cell(add_synapses=False))
				os.chdir(root_path)
				i += 1
		for j in range(len(models) - 1, len(models) - remaining_cells - 1, -1):
			os.chdir(models[j])
			cells.append(eval(models[j].lower()).create_cell(add_synapses=False))
			os.chdir(root_path)

def build_network():
	print('Establishing synapses.')
	for source in cells:
		for target in cells:
			establishSynapses(source, target)

def simAcc():
	gate = dynamicGate() # Remember to change bitwise_and/or and numSlots and numConnections
	# lc = circuit()
	# slotSize = 5
	stim = []
	syn_ = []
	ncstim = []

	stim2 = []
	syn_2 = []
	ncstim2 = []

	# stim3 = []
	# syn_3 = []
	# ncstim3 = []

	# stim4 = []
	# syn_4 = []
	# ncstim4 = []

	# bt1 = poissonFiring(1, 20, 0.5)[0]
	# bt2 = poissonFiring(1, 20, 0.5)[0]

	# bt1 = poissonFiring(1, 20, float(sys.argv[1]))[0]
	# bt2 = poissonFiring(1, 20, float(sys.argv[1]))[0]
	# bt3 = poissonFiring(1, 20, float(sys.argv[1]))[0]
	# bt4 = poissonFiring(1, 20, float(sys.argv[1]))[0]

	global freq

	stim = neuron.h.NetStim()
	syn_ = neuron.h.ExpSyn(0.5, sec=gate.input1)
	stim.start = 0
	stim.number = 1e9
	stim.seed(numpy.random.randint(100, size=1)[0])
	stim.noise = 1
	stim.interval = 1000/freq
	ncstim = neuron.h.NetCon(stim, syn_)
	ncstim.delay = 0
	ncstim.weight[0] = weight
	syn_.tau = 2

	stim2 = neuron.h.NetStim()
	syn_2 = neuron.h.ExpSyn(0.5, sec=gate.input2)
	stim2.start = 0
	stim2.number = 1e9
	stim.seed(numpy.random.randint(100, size=1)[0])
	stim2.noise = 1
	stim2.interval = 1000/freq
	ncstim2 = neuron.h.NetCon(stim2, syn_2)
	ncstim2.delay = 0
	ncstim2.weight[0] = weight
	syn_2.tau = 2

	for cell, idNum in zip(gate.cells, range(len(gate.cells))):
		recordings['{}_{}'.format(cell, idNum)] = neuron.h.Vector()
		timeRecordings['{}_{}'.format(cell, idNum)] = neuron.h.Vector()
		synCurrentRec['{}_{}'.format(cell, idNum)] = neuron.h.Vector()
		# if idNum == 2:
		# 	synCurrentRec['{}_{}'.format(cell, idNum)].record(syns[0]._ref_i)
		# 	for dendId in range(len(cell.dend)):
		# 		dendRecordings['{}_{}_{}'.format(cell, idNum, dendId)] = neuron.h.Vector()
		# 		dendRecordings['{}_{}_{}'.format(cell, idNum, dendId)].record(cell.dend[dendId](0.5)._ref_v)
		recordings['{}_{}'.format(cell, idNum)].record(cell.soma[0](0.5)._ref_v)
		timeRecordings['{}_{}'.format(cell, idNum)].record(neuron.h._ref_t)
	
	print('The simulation has just started.')
	neuron.h.tstop = simTime
	neuron.h.run()

def circuitTest():
	global freq
	# circuit = filterCircuit()
	circuit = circuitD()

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

	seed_rand = numpy.random.randint(1000, size=1)[0]

	stim = neuron.h.NetStim()
	syn_ = neuron.h.ExpSyn(0.5, sec=circuit.in1)
	stim.start = 0
	stim.number = 1e9
	stim.noise = 1
	stim.seed(numpy.random.randint(1000, size=1)[0])
	stim.interval = 1000 / freq
	ncstim = neuron.h.NetCon(stim, syn_)
	ncstim.delay = 0
	ncstim.weight[0] = weight
	syn_.tau = 2

	stim2 = neuron.h.NetStim()
	syn_2 = neuron.h.ExpSyn(0.5, sec=circuit.in2)
	stim2.start = 0
	stim2.number = 1e9
	stim2.noise = 1
	stim2.seed(numpy.random.randint(1000, size=1)[0])
	stim2.interval = 1000 / freq
	ncstim2 = neuron.h.NetCon(stim2, syn_2)
	ncstim2.delay = 0
	ncstim2.weight[0] = weight
	syn_2.tau = 2

	stim3 = neuron.h.NetStim()
	syn_3 = neuron.h.ExpSyn(0.5, sec=circuit.in3)
	stim3.start = 0
	stim3.number = 1e9
	stim3.noise = 1
	stim3.seed(numpy.random.randint(1000, size=1)[0])
	stim3.interval = 1000 / freq
	ncstim3 = neuron.h.NetCon(stim3, syn_3)
	ncstim3.delay = 0
	ncstim3.weight[0] = weight
	syn_3.tau = 2

	stim4 = neuron.h.NetStim()
	syn_4 = neuron.h.ExpSyn(0.5, sec=circuit.in4)
	stim4.start = 0
	stim4.number = 1e9
	stim4.noise = 1
	stim4.seed(numpy.random.randint(1000, size=1)[0])
	stim4.interval = 1000 / freq
	ncstim4 = neuron.h.NetCon(stim4, syn_4)
	ncstim4.delay = 0
	ncstim4.weight[0] = weight
	syn_4.tau = 2

	for cell, idNum in zip(circuit.cells, range(len(circuit.cells))):
		recordings['{}_{}'.format(cell, idNum)] = neuron.h.Vector()
		timeRecordings['{}_{}'.format(cell, idNum)] = neuron.h.Vector()
		recordings['{}_{}'.format(cell, idNum)].record(cell.soma[0](0.5)._ref_v)
		timeRecordings['{}_{}'.format(cell, idNum)].record(neuron.h._ref_t)

	print('The simulation has just started.')
	neuron.h.tstop = simTime
	neuron.h.run()

def gatesTest():
	gate = xor()
	freq = float(sys.argv[1])

	stim = []
	syn_ = []
	ncstim = []

	stim2 = []
	syn_2 = []
	ncstim2 = []

	stim = neuron.h.NetStim()
	syn_ = neuron.h.ExpSyn(0.5, sec=gate.input1)
	stim.start = 0
	stim.number = 1e9
	stim.noise = 1
	stim.seed(numpy.random.randint(10, size=1)[0])
	stim.interval = 1000 / freq
	ncstim = neuron.h.NetCon(stim, syn_)
	ncstim.delay = 0
	ncstim.weight[0] = weight
	syn_.tau = 2

	stim2 = neuron.h.NetStim()
	syn_2 = neuron.h.ExpSyn(0.5, sec=gate.input2)
	stim2.start = 0
	stim2.number = 1e9
	stim2.noise = 1
	stim2.seed(numpy.random.randint(10, size=1)[0])
	stim2.interval = 1000 / freq
	ncstim2 = neuron.h.NetCon(stim2, syn_2)
	ncstim2.delay = 0
	ncstim2.weight[0] = weight
	syn_2.tau = 2

	for cell, idNum in zip(gate.cells, range(len(gate.cells))):
		recordings['{}_{}'.format(cell, idNum)] = neuron.h.Vector()
		timeRecordings['{}_{}'.format(cell, idNum)] = neuron.h.Vector()
		recordings['{}_{}'.format(cell, idNum)].record(cell.soma[0](0.5)._ref_v)
		timeRecordings['{}_{}'.format(cell, idNum)].record(neuron.h._ref_t)

	print('The simulation has just started.')
	neuron.h.tstop = simTime
	neuron.h.run()

def setRecordings():
	""" Set recording vectors for each cell. """
	print('Setting recording vectors.')
	for cell, idNum in zip(cells, range(len(cells))):
		recordings['{}_{}'.format(cell, idNum)] = neuron.h.Vector()
		timeRecordings['{}_{}'.format(cell, idNum)] = neuron.h.Vector()
		recordings['{}_{}'.format(cell, idNum)].record(cell.soma[0](0.5)._ref_v)
		timeRecordings['{}_{}'.format(cell, idNum)].record(neuron.h._ref_t)

def runSimulation():

	print('The simulation has just started.')
	neuron.h.tstop = simTime
	neuron.h.run()

def exportData():
	print('Exporting recorded data.')
	pickle.dump(recordings, open(os.getcwd() + '/Results/' + printTimeStamp('recordings.pkl'), 'wb'))
	# pickle.dump(dendRecordings, open(os.getcwd() + '/Results/' + printTimeStamp('dendRecordings.pkl'), 'wb'))
	# pickle.dump(synCurrentRec, open(os.getcwd() + '/Results/' + printTimeStamp('synCurrentRec.pkl'), 'wb'))
	# print('Nothing was exported.')

getAllModels()
loadSimulationParameters()

# createCells()
# gatesTest()
circuitTest()

# simAcc()

# build_network()
# setParameters()
# setRecordings()
# runSimulation()

exportData()
# plotOutput(recordings)
getSpikesAsBits()

