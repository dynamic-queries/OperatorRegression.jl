module WaveSurrogates

    using OrdinaryDiffEq
    using LinearAlgebra
    using Distributions
    using Flux
    using NeuralOperators
    using UnPack
    using Random

    include("DataGen.jl")
    export FinteDiff, Spectral, PseudoSpectral
    export OneD, TwoD, ThreeD
    export Grid
    export SquaredExponential
    export GP

    include("Solvers.jl")
    export acousticWE

    include("EnsembleProblem.jl")
    export EnsembleProblem
    export solve, munge
    
    include("DeepONet/DeepONet.jl")
    export DeepOpNet, learn
end