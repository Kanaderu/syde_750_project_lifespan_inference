# Learning probability distributions for lifespan inference
**Project for the UWaterloo course SYDE 750**

![Probability distribution representation and learning network overview](https://raw.githubusercontent.com/astoeckel/syde_750_project_lifespan_inference/master/doc/media/diag_px.svg)

## Abstract

Psychological research by Griffiths et al. suggests that humans are capable of near-optimal Bayesian inference for high-level cognitive tasks. In this report, I describe and analyze a simple neural architecture capable of learning a prior probability distribution from empirical measures and performing statistical inference. The proposed network architecture represents probability distributions as a sum of weighted, near-orthogonal, and non-negative basis functions. Weights are learned with the Prescribed Error Sensitivity (PES) rule. Inference is performed by changing the underlying function space followed by gradient descent. I compare the model to human data collected on the Lifespan Inference Task proposed by Griffiths et al.~and discuss strengths and weaknesses of the model.

## Contents of this repository

This repository contains the project report as well as two presentations. You can find those, including the generated PDFs in the `doc` folder. The presentations rely on XeLaTeX and the `Gentium` and `Noto Emoji` fonts. You can build the presentations with `latexmk`.

All Python code used in the project can be found in the `code` folder. The code relies on `numpy`, `scipy`, `matplotlib` and `nengo`. *Note:* To re-run the experiments you need to download a datafile from the Human Mortality Database. See `code/data/README` for more details.
