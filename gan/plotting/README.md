## Plotting scripts

# chi2.py

Inside the script specify a pattern of numpy files with predictions from running `gan.py` as `fnames`, and "truth data" as another numpy file as `data`.
The script will plot some kind of chisquared test metric for various distributions over time.

# evaluate_model.py

Inside the script, specify the model file of the generator from which to make predictions. Make sure the type of noise is correct. Specify the "truth data" file. Running the script will provide comparison plots of various distributions between generated and truth.


# epochs.py

Specify `fnames` and `data` to be the pattern for predictions over various epochs and the "truth data" file. This will plot the mean and standard deviation (compared to real) over time for various distributions.

# compare_models.py

Specify directory containing history.pkl files and a short name for the model in `modelinfos`. This will plot various statistics vs. epoch number for the specified models.
