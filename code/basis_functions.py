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

def cosine_basis(x):
	'''
	Radial basis function computing computing the cosine in the range from
	-pi/2 to pi/2 and otherwise producing zero.
	'''
	return (x > -np.pi / 2) * (x < np.pi / 2) * np.cos(x)


def gaussian_basis(x):
	'''
	Calculates an unnormalised Gaussian basis.
	'''
	return np.exp(-x**2)


def box_basis(x):
	'''
	Returns one in the range -1 < x < 1, zero otherwise.
	'''
	return (x > -1) * (x < 1)


def find_basis_functions(x_min, x_max, n_basis, f_basis):
	'''
	Produces a list of basis functions from the basis function prototype f_basis
	by rescaling and shifting this funciton n_basis times over the interval
	defined by x_min and x_max.

	x_min: minimum value that should be representable by the set of basis
	functions.
	x_max: maximum value that should be representable by the set of basis
	functions.
	n_basis: the number of basis functions to produce.
	f_basis: the basis function prototype. The function must be positive, 
	monotonous for x > 0 and x < 0, and have its maximum at x = 0.
	'''

	# Basis function centres
	offs = np.linspace(x_min, x_max, n_basis)

	# Scale factor
	scale = (x_max - x_min)

	# Evaluation points
	xs = np.linspace(x_min, x_max, 100 * n_basis)

	# Function which calculates the sum over the basis functions in terms of
	# a spread factor sigma and evaluation points xs
	def f(sigma):
		res = np.zeros(len(xs))
		for i in range(n_basis):
			res = res + f_basis((xs - offs[i]) / (scale * sigma))
		return res

	# Gaussian kernel to be used in the error function.
	# Filtering f with a Gaussian kernel seems to be useful,
	# as the max and min functions react to small amounts of
	# noise in f (e.g. when using box_basis).
	kern = gaussian_basis(np.linspace(-100, 100, len(xs)))
	kern = kern / np.sum(kern)

	# Optimize sigma for a small distance of the minimum and maximum to one.
	def e_tar(sigma):
		ys = np.convolve(f(sigma), kern, "same")
		return ((1 - np.max(ys)) ** 2 + (1 - np.min(ys)) ** 2)
	sigma = scipy.optimize.minimize_scalar(e_tar, bounds=(1e-6, 1 / n_basis), 
		method="bounded", options={"xatol": 1e-9}).x

	# Return the new basis functions
	alpha = np.mean(f(sigma))
	def make_f(i):
		return lambda x: f_basis((x - offs[i]) / (scale * sigma)) / alpha
	return [make_f(i) for i in range(n_basis)]

def basis_function_inner_products(x_min, x_max, f_basis):
	'''
	Calculates an inner product matrix for the given functions.

	x_min: value at which to start the integration of the basis functions.
	x_max: value at which to end the integration of the basis functions.
	f_basis: a list of D function objects.
	returns: a DxD numpy array.
	'''

	# Extend the space over which the inner product is calculated a little
	r = x_max - x_min
	x_min = x_min - r * 0.5
	x_max = x_max + r * 0.5
	xs = np.linspace(x_min, x_max, 1000)

	def inner_product(f1, f2):
		return np.sum(f1(xs) * f2(xs)) / len(xs)

	N = len(f_basis)
	C = np.zeros((N, N))
	for i in range(N):
		for j in range(N):
			C[i, j] = inner_product(f_basis[i], f_basis[j])

	return C


def plot_basis_functions(x_min, x_max, fs, weights=None, ax=None):
	'''
	Helper function which can be used to plot a weighted sum of basis functions
	fs.

	x_min: minimum x value used in the plot.
	x_max: maximum x value used in the plot.
	fs: list of functions that should be plotted.
	weights: weights that should be used. If None is given, just assumes
	uniform weights.
	ax: matplotlib axis objects to plot to. If None is given, a new figure is
	created.
	'''

	# Scale the basis functions uniformly if no weights are given
	if weights is None:
		weights = np.ones(len(fs))

	# If no axis to matplotlib axis to plot to is given, create one!
	if ax is None:
		import matplotlib.pyplot as plt
		ax = plt.figure().gca()

	# Plot the basis functions
	xs = np.linspace(x_min, x_max, 1000)
	total = np.zeros(len(xs))
	for i, f in enumerate(fs):
		ys = weights[i] * f(xs)
		total = total + ys
		ax.plot(xs, ys, 'k', linewidth=0.5)
	ax.plot(xs, total, 'k', linewidth=1)
	return ax


def plot_basis_inner_products(x_min, x_max, fs):
	'''
	Helper function which can be used to plot the inner product of a the basis
	functions and its inverse.

	x_min: minimum x value used in the plot.
	x_max: maximum x value used in the plot.
	fs: list of functions for which the inner product should be computed and
	plotted.
	weights: weights that should be used. If None is given, just assumes
	uniform weights.
	'''

	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(1.75, 1.75))
	ax = fig.gca()
	ax.set_xlabel("$j$")
	ax.set_ylabel("$i$")

	# Calculate the inner product matrix
	C = basis_function_inner_products(x_min, x_max, fs)
	cax = ax.imshow(C, cmap="viridis", vmax=0.05, interpolation="none")
#	cbar = fig.colorbar(cax, orientation='horizontal')
	return fig, ax

def plot_basis_function_transformations(x_min, x_max, fs, fn):
	'''
	Visualises the basis functions, the cummulative basis functions and the
	median estimating basis functions.
	'''

	import matplotlib.pyplot as plt
	import matplotlib.cm
	cmap = matplotlib.cm.get_cmap('viridis')
	colours = np.array(list(map(cmap, np.linspace(0, 1, len(fs))))) * 0.75

	r = x_max - x_min
	xs = np.linspace(x_min - r * 0.5, x_max + r * 0.5, 1000)

	fig1 = plt.figure(figsize=(2.5, 2.5))
	ax1 = fig1.gca()

	fig2 = plt.figure(figsize=(2.5, 2.5))
	ax2 = fig2.gca()

	fig3 = plt.figure(figsize=(2.5, 2.5))
	ax3 = fig3.gca()

	for i, f in enumerate(fs):
		ys1 = f(xs)
		ys1 = ys1 / np.max(ys1)

		ys2 = np.cumsum(ys1)
		ys2 = ys2 / np.max(ys2)

		f_median = lambda x: np.sum((xs <= x) * ys1 - (xs >= x) * ys1)
		ys3 = np.array(list(map(f_median, xs)))
		ys3 = ys3 / np.max(ys3)

		ax1.plot(xs, ys1, color=colours[i])
		ax2.plot(xs, ys2, color=colours[i])
		ax3.plot(xs, ys3, color=colours[i])

	ax1.set_xlim(x_min, x_max)
	ax2.set_xlim(x_min, x_max)
	ax3.set_xlim(x_min, x_max)

	ax1.set_ylim(-1, 1)
	ax2.set_ylim(-1, 1)
	ax3.set_ylim(-1, 1)

	fig1.savefig(fn + "_distr.pdf", format='pdf',
	           bbox_inches='tight', transparent=True)
	fig2.savefig(fn + "_cum.pdf", format='pdf',
	           bbox_inches='tight', transparent=True)
	fig3.savefig(fn + "_median.pdf", format='pdf',
	           bbox_inches='tight', transparent=True)


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	import sys, os

	# Create the output directory if it does not exist
	if not os.path.isdir("./out"):
		os.mkdir("./out")

	N = 10
	ws = None

	def mkfig(y_axis=True):
		fig = plt.figure(figsize=(1.75, 1.35))
		ax = fig.gca()
		ax.set_xlabel("Input value $x$")
		if y_axis:
			ax.set_ylabel("$\sum_i \phi_i(x)$")
		else:
			ax.get_yaxis().set_visible(False)
		ax.set_ylim(0.0, 1.2)
		return fig, ax

	fs = find_basis_functions(0, 100, N, box_basis)
	fig, ax = mkfig(True)
	plot_basis_functions(0, 100, fs, ws, ax)
	fig.savefig("out/basis_box.pdf", format='pdf',
	           bbox_inches='tight', transparent=True)
	fig, _ = plot_basis_inner_products(0, 100, fs)
	fig.savefig("out/basis_box_prod.pdf", format='pdf',
	           bbox_inches='tight', transparent=True)
	plot_basis_function_transformations(0, 100, fs, "out/basis_box")

	fs = find_basis_functions(0, 100, N, gaussian_basis)
	fig, ax = mkfig(False)
	plot_basis_functions(0, 100, fs, ws, ax)
	fig.savefig("out/basis_gaussian.pdf", format='pdf',
	           bbox_inches='tight', transparent=True)
	fig, _ = plot_basis_inner_products(0, 100, fs)
	fig.savefig("out/basis_gaussian_prod.pdf", format='pdf',
	           bbox_inches='tight', transparent=True)
	plot_basis_function_transformations(0, 100, fs, "out/basis_gaussian")

	fs = find_basis_functions(0, 100, N, cosine_basis)
	fig, ax = mkfig(False)
	plot_basis_functions(0, 100, fs, ws, ax)
	fig.savefig("out/basis_cosine.pdf", format='pdf',
	           bbox_inches='tight', transparent=True)
	fig, _ = plot_basis_inner_products(0, 100, fs)
	fig.savefig("out/basis_cosine_prod.pdf", format='pdf',
	           bbox_inches='tight', transparent=True)
	plot_basis_function_transformations(0, 100, fs, "out/basis_cosine")

