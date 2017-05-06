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
import scipy.optimize

import hmd_data

def median_bayesian_estimate(a, b0, b1, db, p_a_b, p_b):
	'''
	Numerically calculates an optimal Bayesian estimate for B using a median estimator in the context of

	           P(A | B) * P(B)
	P(B | A) = ---------------
	                P(A)

	a: the value for a
	b0: minimum possible value for b
	b1: maximum possible value for b
	db: delta that should be used for the calculation of the numerical calculation of the cumulative distribution.
	p_a_b: function calculating P(A | B), must take two arguments.
	p_b: function calculating P(B)
	'''

	# Denormalised posterior
	post = lambda b: p_a_b(a, b) * p_b(b)

	# Cummulative posterior distribution difference
	cum_post = lambda bu: (
		 np.sum(list(map(post, np.arange(b0, bu, db)))) -
		 np.sum(list(map(post, np.arange(bu, b1, db))))) * db

	# Estimate the median by searching the minimum of the cum_post function
	return scipy.optimize.minimize_scalar(
		lambda x: abs(cum_post(x)), bounds=(b0, b1), method="bounded",
		options={"xatol": 1e-6}).x


def estimate_lifespan(p_ttotal, t, max_age, age_delta):
	'''
	Uses the Bayesian median estimator to calculate the lifespan of a person
	with age t.

	p_ttotal: prior distribution containing the relative frequencies of
	lifespans.
	t: age of the person.
	max_age: maximum age that should be considered.
	age_delta: resolution of the prior distribution.
	'''

	p_t_ttotal = lambda t, ttotal: 0.0 if t >= ttotal else 1.0 / ttotal
	ttotal_est = lambda t: max(t, median_bayesian_estimate(t, t, max_age, 
	                           age_delta, p_t_ttotal, p_ttotal))
	return ttotal_est(t)


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	import sys, os

	if (len(sys.argv) != 4):
		print("Plots the optimal Bayesian prediction for the lifespan " + 
		      "inference task. Uses the human mortality data from the " + "specified file.")
		print("Usage: ./optimal_prediction.py <FILENAME> <YEAR_START> <YEAR_STOP>")
		sys.exit(1)

	# Create the output directory if it does not exist
	if not os.path.isdir("./out"):
		os.mkdir("./out")

	# Load the distribution from the specified file
	p_ttotal = hmd_data.read_from_file(open(sys.argv[1]),
	                                   int(sys.argv[2]), int(sys.argv[3]))

	# Data range
	min_age = 0
	max_age = 110
	age_delta = 1
	xs = np.arange(min_age, max_age, age_delta)

	# Plot the mortality ground truth
	fig = plt.figure(figsize=(3, 2))
	ax = fig.gca()
	ax.bar(xs, list(map(p_ttotal, xs)), 1, color="#AA305C", linewidth=0.5)
	ax.set_xlabel("Lifespan $t_\mathrm{total}$")
	ax.set_ylabel("Relative frequency $p(t_\mathrm{total})$")
	ax.set_title("Ground truth mortality ({}-{})".format(
		int(sys.argv[2]), int(sys.argv[3])))
	ax.set_xlim(min_age, max_age)
	fig.savefig("out/mortality_ground_truth.pdf", format='pdf', bbox_inches='tight', transparent=True)

	# Psychological data from the Griffiths and Tennenbaum 2006 paper:
	# Optimal Predictions in Everyday Cognition.
	pnts = np.array([[18, 75], [40, 75], [60, 78], [83, 91], [97, 99]]).T

	# Plot optimal median estimation
	fig = plt.figure(figsize=(3, 2))
	ax = fig.gca()
	ax.plot(xs, list(map(lambda t: estimate_lifespan(
	        p_ttotal, t, max_age, age_delta), xs)), color="#AA305C", linewidth=2, 
	        label="Optimal Bayesian")
	ax.plot(pnts[0], pnts[1], color="k", linewidth=0, marker="o",
	        label="Human data")
	ax.set_xlabel("Current age $t$")
	ax.set_ylabel("Estimated lifespan $\hat t_\mathrm{total}$")
	ax.set_title("Bayesian estimator vs. human data")
	ax.set_xlim(min_age, max_age)
	ax.legend(loc="best")
	fig.savefig("out/mortality_optimal_bayesian_estimator.pdf", format='pdf',
	            bbox_inches='tight', transparent=True)
