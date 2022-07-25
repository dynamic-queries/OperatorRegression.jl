using WaveSurrogates
using Test

@testset "WaveSurrogates.jl" begin
    include("GP.jl")
    include("DataGen.jl")
    include("EnsembleProblem.jl")
    include("IO.jl")
end
