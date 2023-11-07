# bic-gmm

This PYME plugin uses a sparse grid search to optimize a Gaussian Mixture Model fitted to a 3D pointcloud using the Bayesian Information Criterion. 
For small numbers of components, this can be done in PYME using the `pointcloud.GaussianMixtureModel` reicpe module, however, the stride=1 search for optimal component count is unrealistic for large mixtures, at which point the parallelized grid search implemented here becomes useful.

## Installation
0. Install PYME (see www.python-microscopy.org)
1. clone this repository
2. in terminal or command prompt, with the appropriate python virtual environment / conda environment active, change into the top directory and run `python setup.py develop`

## Use
Example recipe yaml files can be found in the folder recipe_yamls. Open a localization dataset in PYMEVisualize (https://doi.org/10.1038/s41592-021-01165-9), click on the `Recipe` tab, click `Load Recipe`, and select a yaml file to run one of the example recipes.
Alternatively, you can construct your own recipe by clicking `Add Module` and looking for modules under the heading `bic_gmm`

## Implementation
Implementation benefits heavily from sklearn (https://scikit-learn.org, Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.)
