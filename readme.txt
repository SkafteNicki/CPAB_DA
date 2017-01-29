The scripts should be run in the following order:
* generate_theta.py -> 	generates two folder "gen_theta" and "gen_theta_info" with .pkl files 
			contraining the estimated transformation parametrizations and info about which images they were estimated from

* fit_clusters.py -> 	will fit varGMM to the generated theta values. This scrip will generate a folder "cluster_data" which will contain
			.pkl files with the estimated cluster parameters (both processed and non-processed)

* generate_trans.py ->	will from the estimated cluster parameters generate a file called "transformations.pkl" that contains a number of
			presampled transformations

* train_network.py ->	will preform the actually training of the neural network using all the results from the previous scripts

for information about the different settings for each script, write python "script name" -h for help.

The script utils.py contains all utility functions used for the scrips. In particular this contains the
function set_params() which should be edited in the beginning for controlling a lot of the initial parameters
for doing the image alignment. The function generates a file "params.pkl" which needs to be deleted to run
with new settings.
