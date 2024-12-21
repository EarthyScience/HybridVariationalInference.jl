# HybridVariationalInference

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://EarthyScience.github.io/HybridVariationalInference.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://EarthyScience.github.io/HybridVariationalInference.jl/dev/)
[![Build Status](https://github.com/EarthyScience/HybridVariationalInference.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/EarthyScience/HybridVariationalInference.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/EarthyScience/HybridVariationalInference.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/EarthyScience/HybridVariationalInference.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Extending Variational Inference (VI), an approximate bayesian inversion method,
to hybrid models, i.e. models that combine mechanistic and machine-learning parts.

The model inversion, infers parametric approximations of posterior density
of model parameters, by comparing model outputs to uncertain observations. At
the same time, a machine learning model is fit that predicts parameters of these
approximations by covariates.

