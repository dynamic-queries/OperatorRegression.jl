using WaveSurrogates
using HDF5
using Plots
using Flux
using BSON:@save 


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
    filename = "data/1D_Wave_Equation"
    file = h5open(filename)
    a = read(file["a"])
    x = read(file["x"])
    t = read(file["t"])
    u = read(file["u"])
    close(file)
end 


using TSVD
using Plots

I = 20

# Reduce dataset
A,X,T,U = redact(a,x,t,u,I)
# A,X,T,U,b = reduce(a,x,t,u,I,nr)

## Metadata
dims = OneD()
raw_data = (A,X,T,U)
ninstances, inputsize, intersize , outputsize = metadata(raw_data,dims)

# Model
DL = 64
interwidth = 64 
trunk = Chain(Dense(inputsize[1] => DL, relu),
              Dense(DL => DL, relu),
              Dense(DL => DL, relu),
              Dense(DL => DL, relu),
              Dense(DL => interwidth, relu)
            )
dl = 64
branch = Chain(Dense(sum(intersize) => dl, relu),
               Dense(dl => dl, relu),
               Dense(dl => dl, relu),
               Dense(dl => dl, relu),
               Dense(dl => interwidth, relu)
            )


print("Defining Model, Munging Data ... \n")

model = DeepOpNet(trunk, branch, raw_data)
munge!(model,dims)

k = 10000
model.input[1] = model.input[1][:,1:k]
model.input[2] = model.input[2][:,1:k]
model.output = model.output[:,1:k]

print("Learning Model...\n")

validation = learn(model,dims,10,1e-3)
validation = learn(model,dims,40,1e-3)
validation = learn(model,dims,1e3,1e-3)

print("Saving Model \n")
@save "1D-Wave.bson" model

out = model(validation["input"][:,1:10],validation["inter"][:,1:10])
trueout = validation["output"][:,1:10]
scatter(out,label="eval")
scatter!(trueout[1,:],label="truth")
savefig("true_vs_modelled.png")
print("Computing Error \n")
