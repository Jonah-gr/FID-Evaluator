# FID-Evaluator
The FID-Evaluator is a tool for evaluating the quality of synthetic images. 
It calculates the Fréchet Inception Distance (FID) between real and fake images.


# Installation
Clone the repository and install the dependencies using pip:

git clone https://github.com/jgraeve/FID-Evaluator.git

pip install -r requirements.txt


# Usage
The following three steps need to be executed after each other: compute_features, pca, and fid.

1. Compute Features

To compute features from real and fake images, use the compute_features mode:

python main.py --mode compute_features --real /path/to/real/images --fake /path/to/fake/images --device cuda --noise 0.0 0.1 0.2 0.3 0.4 --noise_types gauss blur swirl rectangles salt_and_pepper


This will compute features from the specified real and fake images, with optional noise applied. The computed features will be saved to a pickle file.

2. Perform PCA

To perform PCA on the computed features, use the pca mode:


python main.py --mode pca --n_components 10 25 50 100 200 300


This will perform PCA on the computed features with the specified number of components. The transformed features will be saved back to the pickle file.

3. Calculate FID
   
To calculate the FID score, use the fid mode:


python main.py --mode fid


This will load the features from the pickle file and calculate the FID score for different noise types and levels. 
The plots will show the percentage increase of the FID scores and not the FID itself.
