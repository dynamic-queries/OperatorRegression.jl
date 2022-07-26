using WaveSurrogates
using HDF5
using Plots
using Flux



print("1D Wave Equation using DeepOpNet.\n\n")

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

function reduce(a,x,t,u,I,nr)
    A = a[:,1:I]
    X = x[:,1:I]
    T = t[:,1:I]
    U = u[:,:,1:I]
    b,_,_ = tsvd(A,nr)

    A = b' * A 
    X = b' * X
    U = reshape(b' * reshape(U,(size(a,1),:)),(:,size(t,1),I)) 

    A,X,T,U,b
end 

function redact(a,x,t,u,I)
    A = a[:,1:I]
    X = x[:,1:I]
    T = t[:,1:I]
    U = u[:,:,1:I]

    A,X,T,U
end 

print("Reading data ...\n")

begin # Read data
    filename = "/home/dynamic-queries/.julia/dev/WaveSurrogates.jl/data/End_1D_Wave_Equation"
    file = h5open(filename)
    a = read(file["a"])
    x = read(file["x"])
    t = read(file["t"])
    u = read(file["u"])
    close(file)
end 


using TSVD
using Plots

I = 1

# Reduce dataset
A,X,T,U = redact(a,x,t,u,I)
# A,X,T,U,b = reduce(a,x,t,u,I,nr)


## Metadata
dims = OneD()
raw_data = (A,X,T,U)
ninstances, inputsize, intersize , outputsize = metadata(raw_data,dims)

size(A)


# Model
DL = 128
interwidth = 128 
trunk = Chain(Dense(inputsize[1] => DL, relu),
              Dense(DL => DL, relu),
              Dense(DL => DL, relu),
              Dense(DL => DL, relu),
              Dense(DL => interwidth, relu)
            )
dl = 32
branch = Chain(Dense(sum(intersize) => dl, relu),
               Dense(dl => dl, relu),
               Dense(dl => dl, relu),
               Dense(dl => dl, relu),
               Dense(dl => interwidth, relu)
            )


print("Defining Model, Munging Data ... \n")

model = DeepOpNet(trunk, branch, raw_data)
munge!(model,dims)

print("Learning Model...\n")

validation = learn(model,dims,10,1e-5)
validation = learn(model,dims,40,1e-5)
validation = learn(model,dims,1e3,1e-5)
validation = learn(model,dims,1e3,1e-5)


using LinearAlgebra

error = validation["output"] - reshape(model(validation["input"],validation["inter"]),(1,:))
scatter(error[1,:])

norm(error)