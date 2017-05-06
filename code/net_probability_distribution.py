#!/usr/bin/env python3

# Nengo Statistical Inference Implementation for the UWaterloo course SYDE 750
# Copyright (C) 2017 Andreas Stöckel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import nengo
import numpy as np

import basis_functions

class ProbabilityDistribution(nengo.Network):
	'''
	ProbabilityDistribution is a subcomponent of the LifespanInference network.
	The network is capable of sampling and learning the empirical distribution
	of an input $x$ and to calculate a median estimating gradient given a
	likelihood function $p(x | y)$.
	'''

	# Strength of the inhibitory connections used to silence populations.
	inhibitory_strength = 2

	@staticmethod
	def build_ensemble_for_range(n_neurons, x_min, x_max, dimensions=1,
							  label=None, seed=None):
		'''
		Used internally to build an ensemble which covers a certain range from
		x_min to x_max. Currently just sets the radius such that the range is
		covered. Later implementations might actually select better gains and
		biases such that the tuning curves better represent the hypercube.
		'''
		radius = max(abs(x_min), abs(x_max)) * np.sqrt(dimensions)
		return nengo.Ensemble(
			n_neurons=n_neurons,
			dimensions=dimensions,
			radius=radius,
			label=label,
			seed=seed)

	@staticmethod
	def make_median_gradient_component(f_likelihood, f_basis, x_min, x_max, 
		mul_conditional):
		'''
		Calculates the alternative basis function Phi(x, y) used to calculate
		the median gradient.

		f_likelihood: likelihood function.
		f_basis: basis function.
		x_min: minimum input value for x.
		x_max: maximum input value for x.
		mul_conditional: 1 for the positive branch of the function, -1 for the
		negative brancht of the function.
		'''
		r = (x_max - x_min)
		xs = np.linspace(x_min - r * 0.05, x_max + r * 0.05, 1000)
		dx = xs[1] - xs[0]
		fxs = f_basis(xs)
		scale = 6.0 # Approximately scale the output to the range of 0 to 1
		def Phi(v):
			x, y = v
			s1 = np.sum(f_likelihood(y, xs) * fxs * (xs <= x) * dx)
			s2 = np.sum(f_likelihood(y, xs) * fxs * (xs >= x) * dx)
			return min(1, max(0, (s1 - s2) * mul_conditional) * scale) # Clamp

		return Phi

	def __init__(self,
	             n_input_neurons=200,
	             n_2d_input_neurons=1000,
	             n_output_neurons=200,
	             n_basis_neurons=20,
	             x_min=-1,
	             x_max=1,
	             n_basis=10,
	             f_basis=basis_functions.cosine_basis,
	             f_likelihood=None,
	             mul_conditional=1,
	             learning_rate=1e-5,
	             record_learning=False,
	             record_dt=10e-3,
	             seed=None,
	             **kwargs):
		'''
		Constructor of the ProbabilityDistribution network class. This function
		constructs a Nengo network model capable of learning a prior probability
		distribution p(x) and to compute a median gradient function for a
		posterior distribution.

		n_input_neurons: number of neurons in the one-dimensional input
		ensemble representing x.
		n_2d_input_neurons: number of neurons in the one-dimensional input
		ensemble representing x and y.
		n_output_neurons: number of neurons in the output population.
		n_basis_neurons: number of neurons in each population representing a
		basis function or an error signal.
		x_min: minimum value for the x input domain. Currently, this value is
		also used for y.
		x_max: maximum value for the y input domain. Currently, this value is
		also used for x.
		n_basis: number of basis functions.
		f_basis: basis function template. Must be a one-dimensional function
		for which the following conditions hold:
			f(0) = 1, f(x) = f(-x), 0 <= f(x) <= 1
		f_likelihood: two-dimensional function representing the likelihood.
		Should be a probability distribution, so it must hold
			0 <= f(x, y) <= 1 and int f(x, y) dy = 1 for all x.
		mul_conditional: Value which is either 1 or -1 and determines which
		branch of the conditional posterior distribution is being computed.
		learning_rate: learning rate kappa used in the PES rule.
		record_learning: if True, records the decoders in order to analyze the
		learning process lateron.
		seed: seed to be used for the ensembles.
		'''

		# Call the inherited constructor and pass all unhandled keyword
		# arguments to it.
		super().__init__(**kwargs)

		# If no seed is specified, generate a random one
		if seed is None:
			seed = np.random.randint(np.iinfo(np.int32).max)

		# Transformations used for inhibitory connections
		trafo_inh_input = (-self.inhibitory_strength *
					 np.ones((n_input_neurons, 1)))
		trafo_inh_input_2d = (-self.inhibitory_strength *
					 np.ones((n_2d_input_neurons, 1)))
		trafo_inh_basis = (-self.inhibitory_strength *
					 np.ones((n_basis_neurons, 1)))

		# Copy some of the parameters, which allows them to be accessed later
		self.x_min = x_min
		self.x_max = x_max
		self.n_basis = n_basis
		self.record_learning = record_learning
		self.record_dt = record_dt

		# Create the basis function objects
		self.basis_functions = basis_functions.find_basis_functions(
			x_min, x_max, n_basis, f_basis)

		with self:
			# Create the input nodes
			self.x = nengo.Node(size_in=1, label="x")
			self.output = nengo.Node(size_in=1, label="fx")
			self.stop_learning = nengo.Node(size_in=1, label="stop_learning")
			self.stop_f1 = nengo.Node(size_in=1, label="stop_f1")

			# Create the input ensemble, connect the input node and the stop_f1
			# nodes to the input node.
			self.ens_input_f1 = self.build_ensemble_for_range(
				n_input_neurons, x_min, x_max, dimensions=1,
				label="ens_input_f1", seed=seed)
			nengo.Connection(self.x, self.ens_input_f1)
			nengo.Connection(
				self.stop_f1,
				self.ens_input_f1.neurons,
				transform=trafo_inh_input)

			# Create some additional nodes if a likelihood function is given
			if not f_likelihood is None:
				self.stop_f2 = nengo.Node(size_in=1, label="stop_f2")
				self.y = nengo.Node(size_in=1, label="y")
				self.ens_input_f2 = self.build_ensemble_for_range(
					n_2d_input_neurons, x_min, x_max, dimensions=2,
					label="ens_input_f2", seed=seed)
				nengo.Connection(self.x, self.ens_input_f2[0])
				nengo.Connection(self.y, self.ens_input_f2[1])
				nengo.Connection(
					self.stop_f2,
					self.ens_input_f2.neurons,
					transform=trafo_inh_input_2d)

			# Create the output ensembles, connect the ensembles to the output
			# nodes
			self.ens_output = nengo.Ensemble(
				n_neurons=n_output_neurons,
				dimensions=1,
				label="ens_output",
				seed=seed,
				max_rates=nengo.dists.Uniform(100, 200))
			nengo.Connection(self.ens_output, self.output)

			def make_basis_array(label, dir):
				ea = nengo.networks.EnsembleArray(
					n_basis_neurons,
					n_basis,
					label=label,
					encoders=nengo.dists.Uniform(dir, dir),
					intercepts=nengo.dists.Uniform(0.0, 0.95),
					max_rates=nengo.dists.Uniform(100, 200),
				)
				for i, ens in enumerate(ea.ea_ensembles):
					ens.seed = (seed * i + 319 * i) % np.iinfo(np.int32).max
				return ea

			self.ens_basis = make_basis_array("ens_basis",  1)
			self.ens_error = make_basis_array("ens_error", -1)
			self.learning_connections = [None] * n_basis
			self.learning_connection_probes = [None] * n_basis
			for i, f in enumerate(self.basis_functions):
				# Decode the basis functions from the input
				nengo.Connection(self.ens_input_f1,
					self.ens_basis.ea_ensembles[i], function=f)
				if f_likelihood:
					nengo.Connection(
						self.ens_input_f2,
						self.ens_basis.ea_ensembles[i],
						function=self.make_median_gradient_component(
							f_likelihood, f, x_min, x_max, mul_conditional),
						eval_points=np.random.uniform(x_min, x_max, (2000, 2)),
						scale_eval_points=False)

				# Sum the basis function activities in the output, start with
				# a connection producing "zero"
				self.learning_connections[i] = nengo.Connection(
					self.ens_basis.ea_ensembles[i],
					self.ens_output, transform=0)
				self.learning_connections[i].learning_rule_type = nengo.PES(
					learning_rate=learning_rate)
				if record_learning:
					self.learning_connection_probes[i] = nengo.Probe(
						self.learning_connections[i], 'weights', record_dt,
						synapse=None)

				# Mirror the current activity to the error ensemble, use
				# the error as an input to the learning rule
				nengo.Connection(
					self.ens_basis.ea_ensembles[i],
					self.ens_error.ea_ensembles[i],
					function=lambda x: x ** 2,
					transform=-1)
				nengo.Connection(
					self.ens_error.ea_ensembles[i],
					self.learning_connections[i].learning_rule)

				# Connect the stop_learning inhibitory input node to the
				# error ensemble
				nengo.Connection(self.stop_learning, 
					self.ens_error.ea_ensembles[i].neurons,
					transform=trafo_inh_basis)

	def retrieve_recorded_weights(self, sim):
		'''
		Reconstructs the weight matrix for the underlying mixture model from the
		decoders recorded over time.

		sim: simulator object from which holds the probed decoders.
		'''

		from nengo.utils.ensemble import tuning_curves

		# Make sure data was recorded
		assert(self.record_learning)

		# Fetch the decoders over time for each basis ensemble
		decoders = [sim.data[probe] for probe in self.learning_connection_probes]

		# Fetch the tuning curves for each basis ensemble
		xs = np.linspace(0, 1, 100).reshape(-1, 1)
		activities = [tuning_curves(ens, sim, xs)[1] for ens in 
				self.ens_basis.ea_ensembles]

		# Calculate the weights
		N = decoders[0].shape[0] # Number of timesteps
		ws = np.zeros((N, self.n_basis))
		for i in range(N): # Iterate over each timestep
			for j in range(self.n_basis): # Iterate over each basis
				# Decode the current function that is being computed
				fs = activities[j]
				dec = decoders[j][i]
				f = fs @ dec.T

				# Obtain the weight value by fitting a linear function to the
				# decoded function
				ws[i, j] = np.polyfit(xs[:, 0], f[:, 0], 1)[0]
		return ws
