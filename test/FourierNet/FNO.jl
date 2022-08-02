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
input[1,:,:] = a[:,:]
input[2,:,:] = x[:,:]
output[1,:,:] = u[:,:]

# Make sure munging is correct
idx = 13
plot(input[1,:,idx])
plot!(output[1,:,idx])

# Setup the FNO net
x = input
y = output

train_data,test_data  = Flux.splitobs((x,y),at=0.90)
train_loader = Flux.DataLoader(train_data,batchsize=10,shuffle=true)
test_loader = Flux.DataLoader(test_data,batchsize=10,shuffle=false)


dls = [32,64,128,256,512,1024]
for DL in dls 
        nmodes = 16
        model = Chain(
                Dense(2,DL),
                OperatorKernel(DL=>DL, (nmodes,), FourierTransform, gelu),
                OperatorKernel(DL=>DL, (nmodes,), FourierTransform, gelu),
                OperatorKernel(DL=>DL, (nmodes,), FourierTransform, gelu),
                OperatorKernel(DL=>DL, (nmodes,), FourierTransform, gelu),
                Dense(DL,DL),
                Dense(DL,1)
        )

        opt = Flux.Optimiser(WeightDecay(1e-4),Flux.ADAM(1e-3))
        lossfunction = l₂loss
        data = (train_loader,test_loader)

        learner = Learner(model,data, opt,lossfunction,Checkpointer(joinpath(@__DIR__,"checkpoints/$(DL)/")))
        fit!(learner,1e3)

        # Validate model 
        iter = [1,10,25,30,50,60,75,80,100]

        for i in iter
                xg = input[2,:,i]
                ag = input[1,:,i]
                ug = output[1,:,i]
                val_i = input[:,:,i]
                trial = model(reshape(val_i,(2,1001,1)))[1,:,1]
                plot(xg,trial,label="trial",xlabel="Grid - x",ylabel="Amplitude of Oscillations")
                display(plot!(xg,ug,label="actual",title="$(DL) - Depth of Layers"))
                savefig(joinpath(@__DIR__,"figures/$(DL)_$(i).png"))
        end 
end 