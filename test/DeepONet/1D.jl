using WaveSurrogates
using HDF5
using Plots
using Flux

function metadata(raw_data,::OneD)
    a,x,t,u = raw_data
    @assert size(a,2) == size(x,2) == size(t,2) == size(u,3)
    I = size(a,2)
    @assert size(a,1) == size(x,1)
    inputsize = [size(a,1),1]
    intersize = [1,1]
    outputsize = 1
    I,inputsize,intersize,outputsize
end 

function metadata(raw_data,::TwoD)
    a,x,t,u = raw_data
    
end 

begin # Read data
    filename = "/home/dynamic-queries/.julia/dev/WaveSurrogates.jl/data/End_1D_Wave_Equation"
    file = h5open(filename)
    a = read(file["a"])
    x = read(file["x"])
    t = read(file["t"])
    u = read(file["u"])
    close(file)
end 


# Reduce dataset
I = 10
A = a[:,1:I]
X = x[:,1:I]
T = t[:,1:I]
U = u[:,:,1:I]

## Metadata
dims = OneD()
raw_data = (A,X,T,U)
ninstances, inputsize, intersize , outputsize = metadata(raw_data,dims)

# Model
DL = 20
interwidth = 40 
trunk = Chain(Dense(inputsize[1] => DL, gelu),
              Dense(DL => DL, gelu),
              Dense(DL => DL, gelu),
              Dense(DL => DL, gelu),
              Dense(DL => interwidth, gelu)
            )
dl = 20
branch = Chain(Dense(sum(intersize) => dl, gelu),
               Dense(dl => dl, gelu),
               Dense(dl => dl, gelu),
               Dense(dl => dl, gelu),
               Dense(dl => interwidth, gelu)
            )


model = DeepOpNet(trunk, branch, raw_data)
munge!(model,dims)
model.input[1]
model.input[2]
model.output

learn(model,dims,100)