using NPZ
using Plots

filename = "/home/dynamic-queries/.julia/dev/WaveSurrogates.jl/test/SampleData/Advection_inputs.npy"
a = npzread(filename)
state_size = size(a,1)
nsamples = size(a,2)

filename = "/home/dynamic-queries/.julia/dev/WaveSurrogates.jl/test/SampleData/Advection_outputs.npy"
u = npzread(filename)

t = [0.0,1.0]

@assert state_size == size(u,1)
@assert nsamples == size(u,2)