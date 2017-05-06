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
import matplotlib.pyplot as plt
import sys, os, re

if len(sys.argv) == 1:
	print("""Program for the statistical analysis of the net_probability_*_err.csv
files produced by the analyse_net_probability_distribution.py script.
Produces individual plots for each encountered basis function type.""")
	print("Usage: ./analyse_net_probability_distribution_err.py <CSV FILE 1...> <CSV FILE 2...> ... <CSV FILE n...>")
	sys.exit(1)

# Create the output directory if it does not exist
if not os.path.isdir("./out"):
	os.mkdir("./out")

# Restore the parameters from the filename
fn_re = re.compile(r".*_(box|cosine|gaussian)_([0-9]+)_([0-9]+)_([0-9]+)_err.csv")
fns = sys.argv[1:]
categorised_files = {}
for fn in fns:
	# Fetch some metadata from the files
	matches = fn_re.match(fn)
	if matches is None:
		print("Invalid filename " + fn + ". Make sure to use files produced by ./analyse_net_probability_distribution.py")
		sys.exit(1)
	basis, n_basis, _, i = matches.groups()
	n_basis = int(n_basis)
	i = int(i)

	# Sort the files by basis, number of basis functions and the index i
	if not basis in categorised_files:
		categorised_files[basis] = {}
	if not n_basis in categorised_files[basis]:
		categorised_files[basis][n_basis] = []
	categorised_files[basis][n_basis].append(fn)

for basis, n_basis_dict in categorised_files.items():
	# Fetch the variable describing the number of basis functions $k$ and
	# sort it. For each basis function count load the corresponding files and
	# calculate the mean and standard deviation for the last 10% of values in
	# the file
	ks = list(n_basis_dict.keys())
	ks.sort()

	err_means = np.zeros((len(ks), 3))
	err_stddev = np.zeros((len(ks), 3))

	for i, k in enumerate(ks):
		fns = categorised_files[basis][k]
		accu = []
		for fn in fns:
			data = np.loadtxt(fn, delimiter=",")[:,1:4]
			n = int(data.shape[0] * 0.1)
			accu.append(data[-n:])
		accu = np.concatenate(accu)
		err_means[i, :] = np.mean(accu, axis=0)
		err_stddev[i, :] = np.sqrt(np.var(accu, axis=0))

	fig = plt.figure(figsize=(1.75, 1.5))
	ax = fig.gca()
	ax.errorbar(ks, err_means[:, 0],
		fmt="o", color="k", linestyle=(0, (2, 1)), linewidth=0.5, markersize=3)
	ax.errorbar(ks, err_means[:, 1], err_stddev[:, 1],
		fmt="x-", color="k", markersize=3, linewidth=0.5)
	ax.set_ylim(0, 0.601)
	ax.set_xlim(min(ks) - 2, max(ks) + 2)
	ax.set_xticks(ks)
	ax.set_xlabel("Basis size $k$")
	fig.savefig(
		"out/net_probability_{}_errs.pdf".format(basis),
		format='pdf',
		bbox_inches='tight',
		transparent=True)

