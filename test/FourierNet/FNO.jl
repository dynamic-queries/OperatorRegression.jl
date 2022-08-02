using NeuralOperators
using HDF5
using Plots
using Flux
using FluxTraining
using Flux:Dense

# Weed out data
filename = "data/End_1D_Wave_Equation"
file = h5open(filename)
a,x,t,u = read(file["a"]),read(file["x"]),read(file["t"]),read(file["u"])
u = u[:,end,:]

# View solution
plot(a[:,end])
plot!(u[:,end])

input = zeros(2,size(u,1),100)
output = zeros(1,size(u,1),100)
input[1,:,:] = u[:,100]
input[2,:,:] = x[:,100]
output[1,:,:] = u[:,100]

# Make sure munging is correct
idx = 13
plot(input[1,:,idx])
plot!(output[1,:,idx])

# Setup the FNO net
num_test = 5
ntrain = 50-num_test

x = input[:,:,1:end-1]
y = output[:,:,2:end]

train_data,test_data  = Flux.splitobs((x,y),at=0.90)

train_loader = Flux.DataLoader(train_data, batchsize=5, shuffle=true)
test_loader = Flux.DataLoader(test_data, batchsize=1, shuffle=false)

DL = 256
model = Chain(
        Dense(1,DL),
        OperatorKernel(DL=>DL, (20,), FourierTransform, gelu),
        OperatorKernel(DL=>DL, (20,), FourierTransform, gelu),
        OperatorKernel(DL=>DL, (20,), FourierTransform, gelu),
        OperatorKernel(DL=>DL, (20,), FourierTransform, gelu),
        Dense(DL,DL),
        Dense(DL,1)
)

train_data[1][:,:,1]


loss(x,y) = Flux.Losses.mse(model(x),y)
optimizer = Flux.Optimiser(WeightDecay(1e-4),Flux.ADAM(1e-3))
data = (train_loader,test_loader)

learner = Learner(model,data,optimizer,loss,Checkpointer(joinpath(@__DIR__,"checkpoints/")))
fit!(learner,200)