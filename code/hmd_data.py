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

def read_from_file(f, year_start=None, year_stop=None):
	'''
	Reads human mortality data form the mortality.org website and returns a
	function representing the data as a normalized probability distribution.
	Allows to filter the data by year, e.g. to return the accumulated data for
	the range from 1990 to 2010.

	f: file handle from which the data is read.
	year_start: first data year that should be taken into account or None if no
	lower bound should be used.
	year_stop: last data year (inclusive) that should be taken into account or
	None if no upper bound should be used.
	'''

	# Read data and header from the file
	header = None
	data = []
	for i, line in enumerate(f):
		line = line.strip()
		if (i == 0) or (len(line) == 0): # Skip the first line and empty lines
			continue
		columns = line.split()
		if header is None:
			header = list(map(str.lower, columns))
		else:
			data.append(tuple(map(
				lambda s: float(s[0:-1]) if s.endswith("+") else float(s),
				columns)))

	# Feed all the data into a numpy structured array
	data = np.array(data, dtype=list(map(lambda s: (s, ">f8"), header)))

	# Filter all the entries according to the given range of years
	valid = np.array([True] * len(data), dtype=np.bool)
	if not year_start is None:
		valid = np.logical_and(valid, data["year"] >= year_start)
	if not year_stop is None:
		valid = np.logical_and(valid, data["year"] <= year_stop)
	years = np.unique(data[valid]["year"])
	ages = np.unique(np.sort(data[valid]["age"]))

	# Build the histogram
	hist = np.zeros((2, len(ages)))
	hist[0] = ages
	for i, age in enumerate(ages):
		age_rows = np.logical_and(valid, data["age"] == age)
		hist[1, i] = np.sum(data[age_rows]["total"])

	# Normalize the histogram
	hist[1] = hist[1] / np.sum(hist[1])

	# Actual distribution function returned from this function
	def distr(x):
		if x < min(ages) or x > max(ages):
			return 0.0
		return hist[1, (np.abs(hist[0]-x)).argmin()]

	return distr

def make_random_sample_over_time(distr,
								 x_min=0,
								 x_max=100,
								 sample_duration=100e-3,
								 seed=None):
	'''
	Constructs a function which returns a random sample from the given
	distribution over time t. Each sample is held for sample_duration.

	distr: is a probability distribution function p(x) as for example returned
	by the read_from_file function.
	x_min: minimum value for x.
	x_max: maximum value for x.
	sample_duration: time each sample is held in seconds
	seed: seed determining the time each sample is held.
	return: a function f(t) which deterministically associates a sample from
	p(x) with a time t.
	'''

	# If no seed is specified, generate a random one
	max_int = np.iinfo(np.int32).max
	if seed is None:
		seed = np.random.randint(max_int)

	# Sample the given distribution in the range from 0 to 100
	xs = np.linspace(x_min, x_max, 1000)
	ps = list(map(distr, xs))
	ps = ps / np.sum(ps)

	def f(t):
		# Backup the current random number generator state
		old_state = np.random.get_state()

		# Deterministically set the seed
		np.random.seed((seed + int(t / sample_duration)) % max_int)

		# Sample an age from the distribution
		x = np.random.choice(xs, p=ps)
		
		# Restore the random number generator state
		np.random.set_state(old_state)

		return x

	return f

if __name__ == "__main__":
	import sys

	if (len(sys.argv) != 2):
		print("Reads a Human Mortality Database file and outputs the " +
		      "relative frequency of lifespans as CSV.")
		print("Usage: ./hmd_data.py <FILENAME>")
		sys.exit(1)

	with open(sys.argv[1]) as f:
		distr = read_from_file(f)

	for age in range(0, 111):
		print(str(age) + ", " + str(distr(age)))
