#  Latent Block Model for uncovering the relationships between genetic variation and molecular traits

This repository provides functions to jointly analyze summary statistics of genome-wide association studies, for investigating associations between SNPs and multiple molecular traits, such as gene expression levels or protein abundances. This joint analysis relying on a latent block model (LBM) enables the simultaneous detection of cis- and trans-associations while uncovering groups of SNPs that share similar association patterns with groups of molecular traits through a co-clustering framework. To efficiently infer the model parameters, we develop a batch variational expectation-maximization (VEM) algorithm, capable of handling high-dimensional datasets involving hundreds to thousands of molecular traits.

## Description of the files in the repository

1.  generate_data_clusters.py: Functions for generating simulated data and saving them into files containting the results of each GWAS.

2. import_data_from_batch.py: Functions for selecting markers common across all GWAS and constructing batches of data.

3. initialisation.py: Functions for initializing the model parameters.

4. VEM.py: Functions for performing the inference procedure using a variational expectation-maximization algorithm.

5. simulation_analysis.py: Script for performing simulation study. 


