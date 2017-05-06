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
import net_lifespan_inference
import hmd_data

def build_model(distr,
				test_samples,
				n_basis=10,
				f_basis=basis_functions.cosine_basis,
				sample_duration=100e-3,
				reset_duration=100e-3,
				test_sample_duration=5.0,
				learning_rate=2e-5,
				training_time=50.0,
				t_total_bias=50.0,
				t_total_bias_mode="t",
				seed=None):
	'''
	Builds a Nengo model which tests the lifespan inference network.

	distr: probability distribution function over the range from 0 to 100 as
	for example returned by the hmd_data.read_from_file function.
	test_samples: list of ages $t$ for which $t_total$ should be infered by the
	network.
	n_basis: number of basis functions in the ProbabilityDistribution network.
	f_basis: basis function template used in the ProbabilityDistribution
	network.
	sample_duration: duration each sample from distr is held in seconds.
	reset_duration: duration during which the integrator is inhibited.
	test_sample_duration: duration for which each input test sample is held.
	learning_rate: learning rate kappa for the PES rule.
	t_total_bias: value for t_total during inference.
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
		# Create the t node
		def t_in(t):
			idx = (t - training_time) / test_sample_duration
			if idx >= 0 and idx < len(test_samples):
				return test_samples[int(idx)]
			return 0.0
		nd_t = nengo.Node(t_in, label="t")

		# Create the t_total node
		def t_total_in(t):
			if t < training_time:
				return samples(t)
			if t_total_bias_mode == "t":
				return t_in(t)
			return 50
		nd_t_total = nengo.Node(t_total_in, label="t_total")

		# Create the control input nodes
		nd_learn = nengo.Node(lambda t: t < training_time, label="learn")
		T = training_time + len(test_samples) * test_sample_duration
		def reset_input(t):
			rel_t = (t - training_time) % test_sample_duration
			return (t > training_time and rel_t < reset_duration) or (t > T)
		nd_reset = nengo.Node(reset_input, label="reset")

		# Create the probability distribution network
		net_infer = net_lifespan_inference.LifespanInference(
			x_min=x_min,
			x_max=x_max,
			n_basis=n_basis,
			f_basis=f_basis,
			learning_rate=learning_rate,
			seed=seed,
			label="net_infer")

		# Connect the control signals
		nengo.Connection(nd_t_total, net_infer.t_total_in)
		nengo.Connection(nd_t, net_infer.t_in)
		nengo.Connection(nd_learn, net_infer.learn)
		nengo.Connection(nd_reset, net_infer.ens_integrator.neurons,
			transform=net_infer.trafo_inh)

		# Record the inputs t_total and t, as well as the output t_total
		probes = {
			"t_in": nengo.Probe(nd_t, "output"),
			"t_total_in": nengo.Probe(nd_t_total, "output"),
			"t_total_out": nengo.Probe(net_infer.t_total_out, "output", 
				synapse=0.005),
			"learn": nengo.Probe(nd_learn, "output")
		}

	return model, probes, samples

def analyse_model(sim, probes, distr, test_samples, test_sample_duration, 
                  training_time):

	import matplotlib.pyplot as plt
	import optimal_prediction

	def draw_samples_times(t_samples, ax):
		for t in t_samples:
			ax.plot([t, t], [-200, 200], color="k",
				linestyle=(0, (1, 2)), linewidth=0.5, alpha=0.75)

	def draw_optimal_lifespans(lifespans_opt, ax):
		for i, y in enumerate(lifespans_opt):
			t0 = training_time + test_sample_duration * i
			t1 = training_time + test_sample_duration * (i + 1)
			ax.plot([t0, t1], [y, y], color="#4e9a06", linewidth=1,
				label="Optimal result" if i == 0 else None)

	def draw_optimal_lifespans_errs(lifespans_opt, ts, t_total_out, ax):
		for i, y in enumerate(lifespans_opt):
			t0 = training_time + test_sample_duration * (i + 0.9)
			t1 = training_time + test_sample_duration * (i + 1)
			ts_valid = np.logical_and(ts > t0, ts < t1)
			ys = np.ones(int(np.sum(ts_valid))) * y
			ax.fill_between(ts[ts_valid], ys, t_total_out[ts_valid],
				color="#d3d7cf")

	# Fetch the times
	T = training_time + len(test_samples) * test_sample_duration
	T0 = training_time * 0.9
	ts = np.arange(0, sim.time, sim.dt)
	t_samples = np.arange(training_time, T, test_sample_duration)

	# Calculate the optimally infered lifespans
	lifespans_opt = list(map(lambda t: optimal_prediction.estimate_lifespan(
	        distr, t, 110, 1), test_samples))

	# Fetch the recorded data
	t_in = sim.data[probes["t_in"]][:, 0]
	t_total_in = sim.data[probes["t_total_in"]][:, 0]
	t_total_out = sim.data[probes["t_total_out"]][:, 0]
	learn = sim.data[probes["learn"]][:, 0]

	# Fill the data dictionary
	data = {}
	for i, y in enumerate(test_samples):
		t0 = training_time + test_sample_duration * (i + 0.9)
		t1 = training_time + test_sample_duration * (i + 1)
		ts_valid = np.logical_and(ts > t0, ts < t1)
		data[y] = t_total_out[ts_valid]

	# Plot the data
	fig = plt.figure(figsize=(8.4, 5.25))
	ax1 = plt.subplot2grid((7, 1), (0, 0))
	ax2 = plt.subplot2grid((7, 1), (1, 0), rowspan=2)
	ax3 = plt.subplot2grid((7, 1), (3, 0), rowspan=4)

	draw_samples_times(t_samples, ax1)
	draw_samples_times(t_samples, ax2)
	draw_samples_times(t_samples, ax3)

	ax1.plot(ts, learn, "k", linewidth=1)
	ax1.set_ylim(-0.1, 1.1)
	ax1.set_yticks([0, 1])
	ax1.set_xlim(T0, T)
	ax1.set_title(r"Input $\mathit{learn}$")
	ax1.axes.get_xaxis().set_visible(False)

	ax2.plot(ts, t_in, "k", linewidth=0.5)
	ax2.set_ylim(0, 110)
	ax2.set_xlim(T0, T)
	ax2.set_title("Input $t$")
	ax2.axes.get_xaxis().set_visible(False)

	draw_optimal_lifespans_errs(lifespans_opt, ts, t_total_out, ax3)
	ax3.plot(ts, t_total_in, "#204a87", linewidth=0.5, linestyle=(0, (2, 1)), label="Input $\hat t_\mathrm{total}$")
	ax3.plot(ts, t_total_out, "k", linewidth=0.5, label="Output $t_\mathrm{total}$")
	draw_optimal_lifespans(lifespans_opt, ax3)
	ax3.set_ylim(0, 110)
	ax3.set_xlabel("Time $t$ [s]")
	ax3.set_xlim(T0, T)
	ax3.set_title("Output $t_\mathrm{total}$ and input $\hat t_\mathrm{total}$")
	ax3.legend(loc="best", ncol=3)

	fig.tight_layout()

	return fig, data


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
		default="gaussian")
	parser.add_argument("--n-basis",
		type=int,
		help="Number of basis functions",
		default=10)
	parser.add_argument("--t-learn",
		type=float,
		help="Training time in seconds",
		default=50.0)
	parser.add_argument("--test-sample-duration",
		type=float,
		help="Time for which each test sample is held",
		default=5.0)
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
	parser.add_argument("--t-total-bias-mode",
		type=str,
		choices=["const", "t"],
		help="Specifies whether the initial value for gradient descent is kept constant or is set to the current age input $t$.",
		required=True)

	args = parser.parse_args()

	# Create the output directory if it does not exist
	if not os.path.isdir("./out"):
		os.mkdir("./out")

	# Read lifespan ground truth distribution from the given human mortality
	# database file
	with open(args.hmd) as f:
		distr = hmd_data.read_from_file(f, int(args.hmd_start), int(args.hmd_end))

	def run_analysis(i):
		# Seed for reproducible experiments
		seed = 4189 + i

		# Build the network
		test_samples = [10, 18, 30, 40, 50, 60, 83, 90, 97]
		model, probes, samples = build_model(
			distr=distr,
			test_samples = test_samples,
			f_basis={
				"box": basis_functions.box_basis,
				"cosine": basis_functions.cosine_basis,
				"gaussian": basis_functions.gaussian_basis
			}[args.basis],
			n_basis=args.n_basis,
			test_sample_duration=args.test_sample_duration,
			training_time=args.t_learn,
			learning_rate=args.learning_rate,
			t_total_bias_mode=args.t_total_bias_mode,
			seed=seed)

		# Run the network
		import nengo
		T = args.t_learn + len(test_samples) * args.test_sample_duration
		with nengo.Simulator(model) as sim:
			sim.run(T)

		# Analyse and plot the results
		fig, data = analyse_model(
			sim=sim,
			probes=probes,
			distr=distr,
			test_samples=test_samples,
			test_sample_duration=args.test_sample_duration,
			training_time=args.t_learn)

		# Store the results
		fn = "out/net_lifespan_{}_{}_{}_{}".format(
			args.basis,
			args.n_basis,
			args.t_total_bias_mode,
			i)
		fig.savefig(
			fn + ".pdf",
			format='pdf',
			bbox_inches='tight',
			transparent=True)

		return data

	processes = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes)
	data_dicts = pool.map(run_analysis, range(args.repeat))

	# Merge the value arrays into the first data_dict
	assert(args.repeat > 0)
	for i in range(1, args.repeat):
		for t in data_dicts[i].keys():
			if not t in data_dicts[0]:
				data_dicts[0][t] = data_dicts[i][t]
			else:
				data_dicts[0][t] = np.concatenate((
					data_dicts[0][t], data_dicts[i][t]))
	data = data_dicts[0]

	# Calculate the mean and standard deviation from the experiments
	ts = list(sorted(data.keys()))
	ys = [np.mean(data[t]) for t in ts]
	es = [np.sqrt(np.var(data[t])) for t in ts]

	# Psychological data from the Griffiths and Tennenbaum 2006 paper:
	# Optimal Predictions in Everyday Cognition.
	pnts = np.array([[18, 75], [40, 75], [60, 78], [83, 91], [97, 99]]).T

	# Plot the optimal values, the human data and the results.
	import matplotlib.pyplot as plt
	import optimal_prediction
	fig = plt.figure(figsize=(2.5, 1.6))
	ax = fig.gca()
	xs = np.arange(0, 100, 1)
	ax.plot(xs, list(map(lambda t: optimal_prediction.estimate_lifespan(
	        distr, t, 100, 1), xs)), linestyle=(0, (2, 1)), color="#AA305C",
	        linewidth=1, label="Optimal Bayesian")
	ax.plot(pnts[0], pnts[1], color="k", linewidth=0, marker="o",
	        label="Human data", markersize=4,
	        markeredgecolor="k", markerfacecolor='none')
	ax.errorbar(ts, ys, es, fmt="+", color="k", markersize=4, label="Inference network")
	ax.set_xlabel("Current age $t$")
	ax.set_ylabel("Estimated lifespan $t_\mathrm{total}$")
	ax.set_xlim(0, 100)
	fn = "out/net_lifespan_{}_{}_{}_analysis".format(
		args.basis,
		args.n_basis,
		args.t_total_bias_mode)
	fig.savefig(
		fn + ".pdf",
		format='pdf',
		bbox_inches='tight',
		transparent=True)

