using WaveSurrogates
using Test

@testset "WaveSurrogates.jl" begin
    include("GP.jl")
    include("DataGen.jl")
    include("DeepONet/DeepONet.jl")
end
