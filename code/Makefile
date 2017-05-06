all: out/basis_box.pdf \
     out/mortality_ground_truth.pdf \
     out/net_probability_box_10_100000_0.pdf \
     out/net_probability_cosine_10_100000_0.pdf \
     out/net_probability_gaussian_10_100000_0.pdf \
     out/net_probability_box_5_50000_0_err.csv \
     out/net_probability_cosine_5_50000_0_err.csv \
     out/net_probability_gaussian_5_50000_0_err.csv \
     out/net_probability_box_10_50000_0_err.csv \
     out/net_probability_cosine_10_50000_0_err.csv \
     out/net_probability_gaussian_10_50000_0_err.csv \
     out/net_probability_box_15_50000_0_err.csv \
     out/net_probability_cosine_15_50000_0_err.csv \
     out/net_probability_gaussian_15_50000_0_err.csv \
     out/net_probability_box_20_50000_0_err.csv \
     out/net_probability_cosine_20_50000_0_err.csv \
     out/net_probability_gaussian_20_50000_0_err.csv \
     out/net_probability_box_25_50000_0_err.csv \
     out/net_probability_cosine_25_50000_0_err.csv \
     out/net_probability_gaussian_25_50000_0_err.csv \
     out/net_probability_box_errs.pdf \
     out/net_lifespan_gaussian_10_const_analysis.pdf \
     out/net_lifespan_gaussian_10_t_analysis.pdf \


# Stand-alone figures

out/basis_box.pdf: basis_functions.py
	./basis_functions.py

out/mortality_ground_truth.pdf: hmd_data.py optimal_prediction.py
	./optimal_prediction.py data/mortality_hmd_us_1x1.txt 2000 2006

# Long probability distribution training runs

out/net_probability_box_10_100000_0.pdf: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis box --n-basis 10 --t-sim 100.0 --repeat 1

out/net_probability_cosine_10_100000_0.pdf: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis cosine --n-basis 10 --t-sim 100.0 --repeat 1

out/net_probability_gaussian_10_100000_0.pdf: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis gaussian --n-basis 10 --t-sim 100.0 --repeat 1

# Error over basis function count plot

out/net_probability_box_5_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis box --n-basis 5 --t-sim 50.0 --repeat 5

out/net_probability_cosine_5_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis cosine --n-basis 5 --t-sim 50.0 --repeat 5

out/net_probability_gaussian_5_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis gaussian --n-basis 5 --t-sim 50.0 --repeat 5

out/net_probability_box_10_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis box --n-basis 10 --t-sim 50.0 --repeat 5

out/net_probability_cosine_10_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis cosine --n-basis 10 --t-sim 50.0 --repeat 5

out/net_probability_gaussian_10_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis gaussian --n-basis 10 --t-sim 50.0 --repeat 5

out/net_probability_box_15_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis box --n-basis 15 --t-sim 50.0 --repeat 5

out/net_probability_cosine_15_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis cosine --n-basis 15 --t-sim 50.0 --repeat 5

out/net_probability_gaussian_15_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis gaussian --n-basis 15 --t-sim 50.0 --repeat 5

out/net_probability_box_20_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis box --n-basis 20 --t-sim 50.0 --repeat 5

out/net_probability_cosine_20_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis cosine --n-basis 20 --t-sim 50.0 --repeat 5

out/net_probability_gaussian_20_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis gaussian --n-basis 20 --t-sim 50.0 --repeat 5

out/net_probability_box_25_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis box --n-basis 25 --t-sim 50.0 --repeat 5

out/net_probability_cosine_25_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis cosine --n-basis 25 --t-sim 50.0 --repeat 5

out/net_probability_gaussian_25_50000_0_err.csv: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py
	./analyse_net_probability_distribution.py --basis gaussian --n-basis 25 --t-sim 50.0 --repeat 5

# Probability distribution error 
out/net_probability_box_errs.pdf: analyse_net_probability_distribution_err.py out/*_50000_*_err.csv
	./analyse_net_probability_distribution_err.py out/*_50000_*_err.csv

# Lifespan inference task test
out/net_lifespan_gaussian_10_const_analysis.pdf: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py \
    net_lifespan_inference.py analyse_net_lifespan_inference.py optimal_prediction.py
	./analyse_net_lifespan_inference.py --t-total-bias-mode const --repeat 10

out/net_lifespan_gaussian_10_t_analysis.pdf: hmd_data.py basis_functions.py net_probability_distribution.py analyse_net_probability_distribution.py \
    net_lifespan_inference.py analyse_net_lifespan_inference.py optimal_prediction.py
	./analyse_net_lifespan_inference.py --t-total-bias-mode t --repeat 10

