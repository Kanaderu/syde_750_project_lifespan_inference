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

import numpy as np

import basis_functions
import net_probability_distribution
import hmd_data

def build_model(distr,
				n_basis=10,
				f_basis=basis_functions.cosine_basis,
				sample_duration=100e-3,
				learning_rate=2e-5,
				seed=None):
	'''
	Performs a single training analysis run on a ProbabilityDistribution
	network.

	distr: probability distribution function over the range from 0 to 100 as
	for example returned by the hmd_data.read_from_file function.
	n_basis: number of basis functions in the ProbabilityDistribution network.
	f_basis: basis function template used in the ProbabilityDistribution
	network.
	sample_duration: duration each sample from distr is held in seconds.
	learning_rate: learning rate kappa for the PES rule.
	seed: seed determining both the probability distribution sampling and all
	random properties of the network.
	'''

	import nengo

	# If no seed is specified, generate a random one
	max_int = np.iinfo(np.int32).max
	if seed is None:
		seed = np.random.randint(max_int)

	# Some constants
	x_min = 0
	x_max = 100

	# Create the distribution sampling function
	samples = hmd_data.make_random_sample_over_time(
		distr, x_min=x_min, x_max=x_max, sample_duration=sample_duration,
		seed=seed)

	# Assemble the nengo network
	model = nengo.Network()
	with model:
		# Create the actual input node that is being learned
		nd_t_total = nengo.Node(samples, label="t_total")

		# Create the control input nodes
		nd_stop_learning = nengo.Node(lambda _: 0, label="stop_learning")
		nd_stop_f1 = nengo.Node(lambda _: 0, label="stop_f1")

		# Create the probability distribution network
		net_distr = net_probability_distribution.ProbabilityDistribution(
			x_min=x_min,
			x_max=x_max,
			n_basis=n_basis,
			f_basis=f_basis,
			record_learning=True,
			learning_rate=learning_rate,
			seed=seed,
			label="p_dist")

		# Connect the control signals
		nengo.Connection(nd_t_total, net_distr.x)
		nengo.Connection(nd_stop_learning, net_distr.stop_learning)
		nengo.Connection(nd_stop_f1, net_distr.stop_f1)

	return model, net_distr, samples

def calc_optimal_weights(ss, fs, x_min, x_max, dt):
	'''
	Calculates optiomal weights over time according to the mathematical model.

	ss: input samples.
	fs: basis functions.
	x_min: minimum x value of the probability distribution.
	x_max: maximum x value of the probability distribution.
	dt: recording timestep.
	'''
	N = len(ss)
	NB = len(fs)
	Gamma = basis_functions.basis_function_inner_products(x_min, x_max, fs)
	GammaInv = np.linalg.inv(Gamma)
	ws = np.zeros((N, NB))
	for i in range(1, N):
		phi = np.array([f(ss[i]) for f in fs]).reshape(-1, 1)
		dw = GammaInv @ phi * dt
		ws[i] = ws[i - 1] + dw.T
	return ws


def calc_probability_distribution_from_weights(
		ws, fs, x_min, x_max, dx, clamp=False):
	'''
	Calculates optiomal weights over time according to the mathematical model.

	ss: input samples.
	fs: basis functions.
	x_min: minimum x value of the probability distribution.
	x_max: maximum x value of the probability distribution.
	dt: recording timestep.
	'''

	# Calculate the probability distribution over time
	xs = np.arange(x_min, x_max, dx)
	fsx = np.array([f(xs) for f in fs])
	ps = (np.maximum(ws, 0.0) @ fsx).T
	if clamp:
		ps = np.minimum(ps, 1.0)

	# Normalise the probability distribution such that the sum is one
	for i in range(ps.shape[1]):
		norm = np.sum(ps[:, i])
		if norm > 1e-15:
			ps[:, i] /= norm
	return ps

def calc_ks_distance(p1, p2):
	'''
	Calculates the Kolmogorv-Smirnov distance between the two discrete
	probability distributions.
	'''
	return np.max(np.abs(np.cumsum(p1, axis=0) - np.cumsum(p2, axis=0)), axis=0)

def analyse_model(sim, net_distr, distr, samples, T):
	'''
	Analyses the data recorded by the model and draws some pretty pictures.
	Returns a matplotlib figure object.

	sim: Nengo simulator used to return the data probed by the model.
	net_distr: ProbabilityDistribution instance returned by build_model.
	samples: function returning the samples over time.
	T: total runtime.
	dt: timestep that should be used for the analysis.
	'''

	def plot_distr_over_time(ps, x_min, x_max, ax):
		# Renormalise the probability distribution such that the maximum is one
		ps_renorm = np.array(ps)
		for i in range(ps.shape[1]):
			norm = np.max(ps_renorm[:, i])
			if norm > 1e-15:
				ps_renorm[:, i] /= norm

		# Plot the image
		ax.imshow(
			ps_renorm,
			cmap='viridis',
			extent=(0, T, x_min, x_max),
			origin='lower',
			aspect='auto',
			interpolation='bicubic',
			vmin=0,
			vmax=1)
		ax.set_xlabel("Time $t$")
		ax.set_ylabel("Lifespan $t_\mathrm{total}$")

	import matplotlib.pyplot as plt

	# Fetch the input samples over time
	dt = net_distr.record_dt
	ts = np.arange(0, T, dt)
	ss = list(map(samples, ts))
	N = len(ts)

	# Calculate the weights according to the theoretical model
	x_min = net_distr.x_min
	x_max = net_distr.x_max
	fs = net_distr.basis_functions

	print("Calculating ground truth distribution...")
	pgt = np.tile(
		np.array(list(map(distr, np.arange(x_min, x_max, 0.1)))).reshape(-1, 1),
		(1, N))
	pgt = pgt / np.sum(pgt, axis=0)[None, :]

	print("Calculating reference weights...")
	weights_opt = calc_optimal_weights(ss, fs, x_min, x_max, dt)

	print("Calculating reference probability distribution...")
	pdist_opt = calc_probability_distribution_from_weights(
		weights_opt, fs, x_min, x_max, 0.1)

	print("Reconstructing weights from simulation...")
	weights_sim = net_distr.retrieve_recorded_weights(sim)

	print("Calculating simulated probability distributions...")
	pdist_sim = calc_probability_distribution_from_weights(
		weights_sim, fs, x_min, x_max, 0.1, clamp=True)

	print("Calculating error metrics...")
	E_optimal_vs_ground_truth = calc_ks_distance(pgt, pdist_opt)
	E_sim_vs_ground_truth = calc_ks_distance(pgt, pdist_sim)
	E_sim_vs_optimal = calc_ks_distance(pdist_opt, pdist_sim)

	print("Plotting...")
	fig = plt.figure(figsize=(8.4, 5.25))
	ax1 = fig.add_subplot(4, 1, 1)
	ax1.plot(ts, ss, linewidth=0.5, color='black')
	ax1.set_title("Input samples")
	ax1.set_ylabel("Sample $\hat t_\mathrm{total}$")
	ax1.set_ylim(0, 100)
	ax1.axes.get_xaxis().set_visible(False)

	ax2 = fig.add_subplot(4, 1, 2)
	plot_distr_over_time(pdist_opt, x_min, x_max, ax2)
	ax2.axes.get_xaxis().set_visible(False)
	ax2.set_title("Optimally learned distribution $p(t_\mathrm{total})$ over time")

	ax3 = fig.add_subplot(4, 1, 3)
	plot_distr_over_time(pdist_sim, x_min, x_max, ax3)
	ax3.axes.get_xaxis().set_visible(False)
	ax3.set_title("Distribution $p(t_\mathrm{total})$ learned by the neural network over time")

	ax4 = fig.add_subplot(4, 1, 4)
	ax4.plot(ts, E_optimal_vs_ground_truth,
		  color="k", label="Optimal vs. ground truth", linewidth=0.5, linestyle=(0, (2.0, 1.0)))
	ax4.plot(ts, E_sim_vs_ground_truth, label="Network vs. ground truth")
	ax4.plot(ts, E_sim_vs_optimal, label="Network vs. optimal")
	ax4.set_ylabel("KS-distance")
	ax4.set_title("Quantitative error analysis")
	ax4.set_ylim(0, 0.3)
	ax4.set_xlabel("Training time $t$ [s]")
	ax4.legend(loc="best", ncol=3)

	fig.tight_layout()

	return fig, np.concatenate((
		ts.reshape(-1, 1),
		E_optimal_vs_ground_truth.reshape(-1, 1),
		E_sim_vs_ground_truth.reshape(-1, 1),
		E_sim_vs_optimal.reshape(-1, 1)
	), axis=1)

# Read the probability distribution file
if __name__ == "__main__":
	import argparse, sys, os, multiprocessing

	parser = argparse.ArgumentParser(
		description="Probability distribution network analysis.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--basis",
		type=str,
		choices=["box", "cosine", "gaussian"],
		help="Number of basis functions",
		required=True)
	parser.add_argument("--n-basis",
		type=int,
		help="Number of basis functions",
		default=10)
	parser.add_argument("--t-sim",
		type=float,
		help="Simulation time in seconds",
		default=100.0)
	parser.add_argument("--repeat",
		type=int,
		help="Number of repetitions",
		default=1)
	parser.add_argument("--hmd",
		type=str,
		help="Human mortality data file",
		default="data/mortality_hmd_us_1x1.txt")
	parser.add_argument("--hmd-start",
		type=int,
		help="Start year that should be parsed from the HMD file",
		default=2000)
	parser.add_argument("--hmd-end",
		type=int,
		help="Last year (inclusive) that should be parsed from the HMD file",
		default=2006)
	parser.add_argument("--learning-rate",
		type=float,
		help="Learning rate kappa for the PES rule",
		default=2e-5)

	args = parser.parse_args()

	# Create the output directory if it does not exist
	if not os.path.isdir("./out"):
		os.mkdir("./out")

	def run_analysis(i):
		# Seed for reproducible experiments
		seed = 4189 + i

		# Read lifespan ground truth distribution from the given human mortality
		# database file
		with open(args.hmd) as f:
			distr = hmd_data.read_from_file(f, int(args.hmd_start), int(args.hmd_end))

		# Build the network
		model, net_distr, samples = build_model(
			distr=distr,
			f_basis={
				"box": basis_functions.box_basis,
				"cosine": basis_functions.cosine_basis,
				"gaussian": basis_functions.gaussian_basis
			}[args.basis],
			n_basis=args.n_basis,
			learning_rate=args.learning_rate,
			seed=seed)

		# Run the network
		import nengo
		T = args.t_sim
		with nengo.Simulator(model) as sim:
			sim.run(T)

		# Analyse and plot the results
		fig, errs = analyse_model(
			sim=sim,
			net_distr=net_distr,
			distr=distr,
			samples=samples,
			T=T)

		# Store the results
		fn = "out/net_probability_{}_{}_{}_{}".format(
			args.basis,
			args.n_basis,
			int(args.t_sim * 1000),
			i)
		fig.savefig(
			fn + ".pdf",
			format='pdf',
			bbox_inches='tight',
			transparent=True)
		np.savetxt(
			fn + "_err.csv",
			errs,
			delimiter=",",
			header="t,E_optimal_vs_ground_truth,E_sim_vs_ground_truth,E_sim_vs_optimal")

	processes = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes)
	pool.map(run_analysis, range(args.repeat))
