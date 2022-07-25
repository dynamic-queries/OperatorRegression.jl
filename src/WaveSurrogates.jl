module WaveSurrogates

    using OrdinaryDiffEq
    using LinearAlgebra
    using Distributions
    using Flux
    using NeuralOperators
    using UnPack
    using Random
    using Plots
    using HDF5
    using TSVD

    include("DataGen.jl")
    export FiniteDiff, Spectral, PseudoSpectral
    export OneD, TwoD, ThreeD
    export Grid
    export SquaredExponential
    export GP
    export flatten, Array
    export plot, heatmap

    include("Solvers.jl")
    export acousticWE

    include("EnsembleProblem.jl")
    export EnsembleProblem
    export solve, Array, write
    
    include("DeepONet/DeepOpNet.jl")
    export DeepOpNet, munge!, learn

    include("PCANet/PCANet.jl")
    export PCANet, munge!, learn
end