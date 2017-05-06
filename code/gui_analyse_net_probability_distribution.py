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
import net_probability_distribution
import analyse_net_probability_distribution

# Reload some important modules
import imp  
imp.reload(net_probability_distribution)
imp.reload(analyse_net_probability_distribution)

# Load the probability distribution if that has not already been done
if not 'distr' in locals():
	import hmd_data
	with open('data/mortality_hmd_us_1x1.txt') as f:
		distr = hmd_data.read_from_file(f, 2000, 2006)

# Build the model -- can now be visualized by Nengo gui
model, _, _ = analyse_net_probability_distribution.build_model(distr)
