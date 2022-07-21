using WaveSurrogates
using Test

@testset "WaveSurrogates.jl" begin
    include("DataGen.jl")
    include("DeepONet/DeepONet.jl")
end
