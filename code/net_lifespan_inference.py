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
from net_probability_distribution import ProbabilityDistribution

class LifespanInference(nengo.Network):
	'''
	Class implementing the LifespanInference network. This network consists of
	two ProbabilityDistribution networks and an integrator for the median
	estimation.
	'''

	# Strength of the inhibitory connections used to silence populations.
	inhibitory_strength = 2

	def __init__(self,
	             n_1d_neurons=200,
	             n_2d_neurons=1000,
	             n_basis_neurons=20,
	             x_min=0,
	             x_max=100,
	             n_basis=10,
	             f_basis=basis_functions.cosine_basis,
	             learning_rate=1e-5,
	             integrator_tau=100e-3,
	             seed=None,
	             **kwargs):
		'''
		Constructor of the LifespanInference network class. Constructs a Nengo
		model capable of learning a prior probability distribution p(t_total)
		and to perform statistical inference by calculating the median of the
		posterior distribution p(t_total | t) over time.

		n_1d_neurons: number of neurons used to represent scalar value (ages).
		n_2d_neurons: number of neurons used to represent 2d-values.
		n_basis_neurons: number of neurons used to represent the individual
		basis functions of the function space underlying the probability 
		distribution.
		x_min: minimum input value.
		x_max: maximum input value.
		n_basis: number of basis functions.
		f_basis: basis function template.
		learning_rate: learning rate passed to the probability distribution
		networks.
		integrator_tau: integrator time constant in seconds.
		seed: random seed determining the concrete network that is being
		constructed.
		'''

		# Call the inherited constructor and pass all unhandled keyword
		# arguments to it.
		super().__init__(**kwargs)

		# If no seed is specified, generate a random one
		max_int = np.iinfo(np.int32).max
		if seed is None:
			seed = np.random.randint(max_int)

		# Transformations used for inhibitory connections
		trafo_inh_input = self.trafo_inh = (-self.inhibitory_strength *
					 np.ones((n_1d_neurons, 1)))

		# Copy some of the parameters, which allows them to be accessed later
		self.x_min = x_min
		self.x_max = x_max
		self.n_basis = n_basis

		with self:
			# Create the input nodes
			self.t_in = nengo.Node(size_in=1, label="t_in")
			self.t_total_in = nengo.Node(size_in=1, label="t_total_in")
			self.t_total_out = nengo.Node(size_in=1, label="t_total_out")
			self.learn = nengo.Node(size_in=1, label="learn")

			# Derive a negated signal from the learn input
			self.not_learn = nengo.Node(
				lambda _, x: 1 - x,
				size_in=1,
				label="not_learn")

			# Create a buffer ensemble holding t_total
			self.ens_t_total = \
				ProbabilityDistribution.build_ensemble_for_range(
					n_1d_neurons, x_min, x_max,
					seed=(seed * 317) % max_int, label="ens_t_total")

			# Create an integrator for the integration of the median gradient
			self.ens_integrator = \
				ProbabilityDistribution.build_ensemble_for_range(
					n_1d_neurons, x_min, x_max,
					seed=(seed * 317) % max_int, label="ens_integrator")

			# Create the two probability distribution networks
			def f_likelihood(t, t_total):
					return (t <= t_total) / np.maximum(1, t_total)
			def mk_pdist(mul_conditional):
				return ProbabilityDistribution(
					n_input_neurons=n_1d_neurons,
					n_2d_input_neurons=n_2d_neurons,
					n_output_neurons=n_1d_neurons,
					n_basis_neurons=n_basis_neurons,
					x_min=x_min,
					x_max=x_max,
					n_basis=n_basis,
					f_basis=f_basis,
					f_likelihood=f_likelihood,
					mul_conditional=mul_conditional,
					learning_rate=learning_rate,
					seed=seed)
			self.pdist_pos = mk_pdist(1)
			self.pdist_neg = mk_pdist(-1)

			# Wire everything up
			nengo.Connection(self.t_total_in, self.ens_t_total)
			nengo.Connection(self.ens_t_total, self.pdist_pos.x)
			nengo.Connection(self.ens_t_total, self.pdist_neg.x)
			nengo.Connection(self.ens_t_total, self.t_total_out)

			nengo.Connection(self.t_in, self.pdist_pos.y)
			nengo.Connection(self.t_in, self.pdist_neg.y)

			nengo.Connection(self.learn, self.not_learn,
				synapse=None)
			nengo.Connection(self.not_learn, self.pdist_pos.stop_learning,
				synapse=None)
			nengo.Connection(self.not_learn, self.pdist_pos.stop_f1,
				synapse=None)
			nengo.Connection(self.learn, self.pdist_pos.stop_f2)

			nengo.Connection(self.not_learn, self.pdist_neg.stop_learning,
				synapse=None)
			nengo.Connection(self.not_learn, self.pdist_neg.stop_f1,
				synapse=None)
			nengo.Connection(self.learn, self.pdist_neg.stop_f2,
				synapse=None)

			nengo.Connection(self.learn, self.ens_integrator.neurons,
				transform=trafo_inh_input)

			# Wire up the integrator. Scale the input connections up in order to
			# account for the orders of magnitude difference between the
			# probability distribution network output and the age values and to
			# speed up integration (but not by too much, since the system will 
			# start to oscillate).
			input_scale = 2 * (x_max - x_min)
			nengo.Connection(self.ens_integrator, self.ens_t_total)
			nengo.Connection(self.ens_integrator, self.ens_integrator, 
				synapse=integrator_tau)
			nengo.Connection(self.pdist_pos.output, self.ens_integrator,
				transform=-integrator_tau * input_scale)
			nengo.Connection(self.pdist_neg.output, self.ens_integrator,
				transform= integrator_tau * input_scale)

