# rad-diff-POP
Data and code to read in the data from the paper: "Two methods to analyse radial diffusion ensembles: the peril of space- and time- dependent diffusion"

The paper will be linked upon publication; the preprint can be found at https://arxiv.org/abs/2407.04669

# Contents of this repo
1. `RadialDiffusionMetrics.ipynb` A Jupyter notebook that can create all plots used in the paper.
2. `ensemble_functions.py` A Python script needed to generate any smaller simulations (i.e. where ensembles have not been used).

# How to get started

You will also need the corresponding data for the plots from Zenodo, which includes the times to monotonicity found for ensembles (in netcdf, .nc files) and the results of any one-off simulations (.csv files). The one-off simulations (.csv files) can alternatively be generated interactively using the notebook but this is very slow. These cells are commented out in favour of reading in the data instead.


## Instructions
To generate all the plots in one go:
- Download the Jupyter notebook. We recommend opening this in Colab.
- Download the file ensemble_functions.py and place it in the same directory as the notebook.
- Download the data from Zenodo and place in a subdirectory named "data"
- Run all cells. The final two cells create zipped folders, containing all plots and any .csv files (if generated).



