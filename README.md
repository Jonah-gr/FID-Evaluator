# FID-Evaluator
The FID Evaluator is a tool for evaluating the Fréchet Inception Distance (FID). The FID is used to evaluate the quality of synthesized to real images. 
To do this, the feature vectors of the images are first calculated using the [Inception v3](https://en.wikipedia.org/wiki/Inceptionv3) model and then a value is determined using the following formula:
$$d_F\left(\mathcal{N}\left(\mu, \sum\right), \mathcal{N}\left(\mu', \sum'\right)\right)^2 = ||\mu - \mu'||_2^2 + \text{tr}\left(\sum + \sum' - 2\left(\sum\sum'\right)^{\frac{1}{2}}\right).$$

We use a principal component analysis (PCA) to reduce the embedding space in order to better match the dimensional space (2048 dimensions) to the real images.
To evaluate this, we noise the synthesized images and compare the percentage increase starting from the first FID value of the respective dimensional reduction of the embedding space.

# Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
    - [Compute Features](#1-compute-features)
    - [Perform PCA](#2-perform-pca)
    - [Calculate FID](#3-calculate-fid)
3. [Explanation](#explanation)


# Installation
Clone the repository and install the dependencies using pip:

```bash
git clone https://github.com/Jonah-gr/FID-Evaluator.git
cd FID-Evaluator
pip install -r requirements.txt
```

# Usage
The following three steps need to be executed after each other: compute_features, pca, and fid.


## 1. Compute Features

To compute features from real and fake images, use the compute_features mode:

```bash
python -m src.main compute_features -r /path/to/real/images -f /path/to/fake/images --noise "0.25 0.5" --noise_types all
```

| Command | Description | Tip |
| --- | --- | --- |
| -r / --real | Path to the real images | |
| -f / --fake | Path to the fake images | |
| -d / --device | Device to use: cpu or cuda | If no device is specified, cuda is used if it is available, otherwise cpu. |
| --noise | The level of distortion, e.g. "0.25 0.5". | See the differences of the levels [here](#explanation). Default: "0.0 0.1 0.2 0.3 0.4". |
| --noise_types | The type of distortion. | See the differences of the types [here](#explanation). Default: "gauss". If "all", every noise type will be used. With "mix [swirl, rectangles]", the noise types in brackets can be used in this exact order. |

This will compute features from the specified real and fake images, with optional noise applied. The computed features will be saved to a pickle file.


## 2. Perform PCA

To perform PCA on the computed features, use the pca mode:

```bash
python -m src.main pca -n "10 25 50 100 200 300"
```

| Command | Description | Tip |
| --- | --- | --- |
| -n / --n_components | The dimensions to which the feature vectors are reduced | Default: 100 |

This will perform PCA on the computed features with the specified number of components. The transformed features will be saved back to the pickle file.

## 3. Calculate FID
   
To calculate the FID score, use the fid mode:


```bash
python -m src.main fid
```

This will load the features from the pickle file and calculate the FID score for the different noise types and levels. 
The plots will show the percentage increase of the FID scores and not the FID itself.


# Explanation

The feature vectors are saved in a dictionary, which in turn is saved in a pickle file (features.pkl). The same file is extended by the reduced vectors in the pca step. The dictionary should then look something like this:

```python
features_dict = {"real":
                        {"no pca": []},
                        {"pca": 
                                {100: []}},
                "fake": 
                        {"no pca": 
                                {"gauss": 
                                        {0.0: [],
                                        0.25: []},
                                "swirl": 
                                        {0.0: [],
                                        0.25: []}},
                        "pca": 
                                {100:
                                        {"gauss": 
                                                {0.0: [],
                                                0.25: []},
                                        "swirl": 
                                                {0.0: [],
                                                0.25: []}
                                        }
                                }
                        }
                }
```
The respective feature vectors can then be found in the lists.

The FID-Evaluator supports 5 different noise types: 

| noise_type | Description |
| --- | --- |
| "gauss" | Gaussian noise |
| "blur" | Gaussian blur |
| "swirl" | swirled images |
| "rectangles" | implanted black rectangles |
| "salt_and_pepper" | salt and pepper noise |

The graphic below shows the different noise types with noise levels from 0 to 1:
![Figure 1](/public/Figure_1.png)
