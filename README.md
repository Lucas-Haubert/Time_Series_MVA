# Machine Learning for Time Series Project

## Overview

This project is a comprehensive study and application of the paper titled "How to introduce expert feedback in one-class support vector machines for anomaly detection?" by J. Lesouple, C. Baudoin, M. Spigai, J.-Y. Tourneret. The objective is to explore the methodologies and implications of the introduction of an expert feedback in a classification task on time series.

[Read the original paper](https://oatao.univ-toulouse.fr/28058/1/Lesouple_28058.pdf)

## Repository Contents

- `Report.pdf`: Detailed report of the work and findings based on the original paper.
- `Slides.pdf`: Presentation slides summarizing the key points and results of the study.
- `nussvm.py` : Implementation of the article's algorithm.
- `s3vmad.py` : Implementation of a slightly different approach.
- `expert_heartbeat.ipynb`, `preproc_heartbeat.ipynb` and `wavelet_heartbeat.ipynb` : Notebooks that run our work, applied on one dataset, following different pipelines.

## Original Content and Data

The code and methodologies implemented in this project are based on the author's contributions:

[Julien Lesouple's website](http://perso.recherche.enac.fr/~julien.lesouple/)

[Dataset used in the original paper](https://github.com/shubhomoydas/ad_examples/tree/master/ad_examples/datasets/anomaly)

Link for the time series datasets used in the project:

[Time Series Classification Website](https://www.timeseriesclassification.com/dataset.php)

## Grade and acknowledgements

Final grade of the project : 16,5/20

I would like to acknowledge Professor [Laurent Oudre](http://www.laurentoudre.fr/), ENS Paris-Saclay, for his course of Machine Learning for Time Series, part of the MVA master's program.

I also thank [Steven Zheng](https://www.linkedin.com/in/stevenzheng07/), Master MVA, for working with me on this project.
