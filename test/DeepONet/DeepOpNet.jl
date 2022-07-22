struct Random end 
using Flux

function generate_data(::Random,::OneD)
    I = 4
    n = 50
    m = 50
    a = rand(n,I)
    t = rand(m,I)
    x = rand(n,I)
    u = rand(n,m,I)
    (a,x,t,u)
end 

function metadata(raw,::OneD)
    a,x,t,u = raw
    @assert size(a,2) == size(x,2) == size(t,2) == size(u,3)
    npoints = size(a,2)
    inputsize = [size(a,1),2]
    outputsize = 1  
    npoints, inputsize, outputsize 
end 

function generate_data(::Random,::TwoD)
    I = 10
    nx = 20
    ny = 22
    nt = 10
    
    a = rand(nx,ny,I)
    x = rand(nx,ny,I)
    t = rand(nt,I)
    u = rand(nx,ny,nt,I)

    a,x,t,u
end 

function metadata(raw,::TwoD)
    a,x,t,u = raw
    @assert size(a,3) == size(x,3) == size(t,2) == size(u,4)
    npoints = size(a,3)
    inputsize = [(size(a,1),size(a,2)),3]
    outputsize = 1
    npoints,inputsize,outputsize
end 

#----------------------------------------------------#
# 1D problem
# Data
problem = Random()
dims = OneD()
raw_data = generate_data(problem,dims)
npoints, inputsize, outputsize = metadata(raw_data,dims)
intersize = 10

# Trunk 
NL = 5
DL = 512
trunk = Chain(Dense(inputsize[1] => DL, gelu),
              Dense(DL => DL, gelu),
              Dense(DL => DL, gelu),
              Dense(DL => DL, gelu),
              Dense(DL => intersize, gelu))

# Branch 
nl = 5
dl = 512
branch = Chain(Dense(inputsize[2] => dl, gelu),
              Dense(dl => dl, gelu),
              Dense(dl => dl, gelu),
              Dense(dl => dl, gelu),
              Dense(dl => intersize, gelu))

# DeepOpNet
model = DeepOpNet(trunk, branch, raw_data)
munge!(model,dims)
validation_set = learn(model,dims)

# Validate model