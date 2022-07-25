# WaveSurrogates.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dynamic-queries.github.io/WaveSurrogates.jl.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dynamic-queries.github.io/WaveSurrogates.jl.jl/dev/)
[![Build Status](https://github.com/dynamic-queries/WaveSurrogates.jl.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/dynamic-queries/WaveSurrogates.jl.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://travis-ci.com/dynamic-queries/WaveSurrogates.jl.jl.svg?branch=main)](https://travis-ci.com/dynamic-queries/WaveSurrogates.jl.jl)



Is a package for Benchmarking different surrogate models for equations that fall under the umbrella of the Wave Equations based on the paradigm of Operator Regression. This includes features such as: 

## Data generation
- High Performance solvers using Method of lines for the
    - 1D linear, non-linear Acoustic Wave Equation
    - 2D linear, non-linear Acoustic Wave Equation
    - 1D non-linear Elastic Wave Equation
    - 2D non-linear Elastic Wave Equation

- 1D and 2D Gaussian Sampling routines for generating an ensemble of initial conditions.

- A native Julia Type - "Ensemble Problem" that solves the specified initial value problems using the high performance solvers, writing out the generated data,... 

## Operator Regression Algorithms

Operator Regresison encompasses several algorithms that has been proposed in literature. In this package, focus is placed on .\
    - *Fourier Neural Operators* - https://openreview.net/forum?id=c8P9NQVtmnO \
    - *Geo-Fourier Neural Operators* - https://arxiv.org/abs/2207.05209 \
    - *DeepONets* - https://www.nature.com/articles/s42256-021-00302-5 \
    - *PCA Net* - \
    - *PARA Net* - https://arxiv.org/abs/2203.13181 \
    - *Operator Inference* (from the Wilcox group) - https://arxiv.org/abs/1912.08177 \
    - *PDE Net with Diffusion Maps* - https://www.nature.com/articles/s41467-022-30628-6 

## Benchmarks

The cost-accuracy tradeoff of each of these algorithms for the Wave equations(ones specified and more to be added) is evaluated based on the following criteria: 

    - Type of sampling the manifold. 
    - Number of samples 
    - Time for data generation 
    - Time to train 
    - Time for evaluation 
    - Relative error (with conventional solver)

Further metrics may be added in the future for the evaluation.

