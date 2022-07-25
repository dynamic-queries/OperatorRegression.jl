using Flux
using WaveSurrogates
using WaveSurrogates:write

function metadata(raw,::OneD)
    a,x,t,u = raw
    @assert size(a,2) == size(x,2) == size(t,2) == size(u,3)
    npoints = size(a,2)
    inputsize = [size(a,1),2]
    outputsize = 1  
    npoints, inputsize, outputsize 
end 


print("1D Wave Equation : Operator Regression using DeepONet\n")
dims = OneD()
ninstances = 2
nx = 1000
nt = 100
xmin = 0.0
xmax = 1.0
tmin = 0.0
tmax = 1.0 
x = xmin : (xmax-xmin)/nx : xmax
t = tmin : (tmax-tmin)/nt : tmax
method = FiniteDiff()
s = x 
print("Generating Data...\n")
prob = EnsembleProblem(dims,method,s,t,ninstances)
sol = solve(prob,acousticWE)

print("Data Generated! \n")
raw_data = Array(sol,dims)
npoints, inputsize, outputsize = metadata(raw_data,dims)
intersize = 10

print("Writing Data for reference\n")
write(sol,dims)

# Trunk 
NL = 5
DL = 20
trunk = Chain(Dense(inputsize[1] => DL, gelu),
              Dense(DL => DL, gelu),
              Dense(DL => DL, gelu),
              Dense(DL => DL, gelu),
              Dense(DL => intersize, gelu))

# Branch 
nl = 5
dl = 20
branch = Chain(Dense(inputsize[2] => dl, gelu),
              Dense(dl => dl, gelu),
              Dense(dl => dl, gelu),
              Dense(dl => dl, gelu),
              Dense(dl => intersize, gelu))

# DeepOpNet
print("Setting Up Model \n")
model = DeepOpNet(trunk, branch, raw_data)

print("Rearranging Data \n")
munge!(model,dims)

print("Training Model \n")
validation_set = learn(model,dims)