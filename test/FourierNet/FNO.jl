using NeuralOperators
using HDF5
using Plots
using Flux
using FluxTraining
using Flux:Dense

# Weed out data
filename = "WaveSurrogates.jl/data/End_1D_Wave_Equation"
file = h5open(filename)
a,x,t,u = read(file["a"]),read(file["x"]),read(file["t"]),read(file["u"])
u = u[:,end,:]

# View solution
plot(a[:,end])
plot!(u[:,end])


# Write munge function 
required_time_step = 99
timesteps_to_train = collect(100-50:99)

input = zeros(1,size(u,1),length(timesteps_to_train))
output = zeros(1,size(u,1),length(timesteps_to_train))
input[1,:,:] = u[:,timesteps_to_train]
# input[2,:,:] = x[:,timesteps_to_train]
output[1,:,:] = u[:,Int.(timesteps_to_train.+1)]

# Make sure munging is correct
idx = 13
plot(input[1,:,idx])
plot!(output[1,:,idx])

# Setup the FNO net
num_test = 5
ntrain = 50-num_test

x = input[:,:,1:end-1]
y = output[:,:,2:end]

train_data,test_data  = Flux.splitobs((x,y),at=0.95)

train_loader = Flux.DataLoader(train_data, batchsize=47, shuffle=true)
test_loader = Flux.DataLoader(test_data, batchsize=2, shuffle=true)

DL = 64
model = Chain(
        Dense(1,DL),
        OperatorKernel(DL=>DL, (16,), FourierTransform, gelu),
        OperatorKernel(DL=>DL, (16,), FourierTransform, gelu),
        OperatorKernel(DL=>DL, (16,), FourierTransform, gelu),
        OperatorKernel(DL=>DL, (16,), FourierTransform, gelu),
        Dense(DL,128),
        Dense(128,1)
)

train_data[1][:,:,1]


loss(x,y) = Flux.Losses.mse(model(x),y)
optimizer = Flux.Optimiser(WeightDecay(1e-4),Flux.Adam(1e-5))
data = (train_loader,test_loader)

learner = Learner(model,data,optimizer,loss,Checkpointer(joinpath(@__DIR__,"checkpoints/")))
fit!(learner,100)