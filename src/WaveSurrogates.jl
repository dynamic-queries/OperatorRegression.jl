module WaveSurrogates

    using OrdinaryDiffEq
    using LinearAlgebra
    using Distributions
    using Flux
    using NeuralOperators
    using UnPack
    using Random
    using Plots

    include("DataGen.jl")
    export FinteDiff, Spectral, PseudoSpectral
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
    export solve, munge
    
    include("DeepONet/DeepOpNet.jl")
    export DeepOpNet, munge!, learn
end