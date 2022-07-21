""" 
    Index: 
        m   -   Number of sensors <=> Number of points in the discretization of the PDE 
        d   -   Dimensions of the space
        nₜ  -   Number of time points
        a   -   Input function 
        x   -   Grid 
        t   -   Time 
        u   -   Output function
        p   -   Number of layers in the merge layer 

    Types: 
        IT  -   Type of the input 
        OT  -   Type of the output 
        GT  -   Type of the grid
        TT  -   Type of the time 

    Assumptions: 
        a,x,t,u are Arrays
        1st dimension of the inputs refers to the number of datapoints
""" 
mutable struct DeepOpNet
    ## Data
    a
    inter
    u

    ## NN 
    NL::Int
    DL::Int
    p::Int
    learning_rate::Float64
    nepochs::Int
    
    # Trainable data 
    atrain 
    itrain
    utrain

    # Test data
    atest
    itest 
    utest

    function DeepOpNet(a,inter,u)
        DL = 1024
        NL = 5 
        p = 1024
        lr = 0.001
        nepochs = 500
        new(a,inter,u,NL,DL,p,lr,nepochs)
    end 

    function DeepOpNet(a,inter,u,DL,NL,p,lr,epochs)
        new(a,inter,u,NL,DL,p,lr,epochs)
    end 
end 

function Base.show(io::IO, prob::DeepOpNet)
    print(io,"DeepOpNet\n")
    print(io,"-----------------------------\n")
    print(io,"Number of data points = $(size(prob.a,1))")
end 


function shuffle!(a,u,inter)
    l = size(a,1) 
    r = randperm(l)
    a = a[r,:]
    u = u[r]
    inter = inter[r,:]
    nothing
end 

function munge!(dpo::DeepOpNet,κ=0.9)
    @unpack a,u,inter = dpo
    
    ND = size(a)[1]
    ntrain = floor(Int,κ*ND)
    shuffle!(a,u,inter)

    dpo.atrain = a[1:ntrain,:]
    dpo.itrain = inter[1:ntrain,:]
    dpo.utrain = u[1:ntrain,:]

    dpo.atest = a[ntrain+1:end,:]
    dpo.itest = inter[ntrain+1:end,:]
    dpo.utest = u[ntrain+1:end,:]
    nothing
end


function model(op::DeepOpNet)
    @unpack atrain,itrain,NL,DL,p = dponet

    sa = size(atrain)
    si = size(itrain)

    branch = Chain(Dense(sa[2] => DL, gelu), Dense(DL => DL, gelu), Dense(DL => DL, gelu), Dense(DL => p, gelu), Dense(DL => DL, gelu))
    trunk = Chain(Dense(si[2] => DL, gelu),Dense(DL => DL, gelu) , Dense(DL => DL, gelu), Dense(DL => DL, gelu),  Dense(DL => p, gelu))
    model = DeepONet(branch,trunk)
    model
end 

"""
    learn(dponet::DeepOpNet)

    Encapsulates the workflow of training the DeepONet model for the given data. 
    Note that the data has to be supplied in a format specified in DataManager.jl. 
"""
function learn(dponet::DeepOpNet) 
        # Pre-Process data
    munge!(dponet)

    # Data
    @unpack atrain,atest,utrain,utest,itrain,itest = dponet 
    @unpack learning_rate,nepochs = dponet

    atrain = atrain' 
    atest = atest' 
    utrain = utrain' 
    utest = utest' 
    itrain = itrain' 
    itest = itest'


    # Generate a model 
    mod = model(dponet)


    # Optimize the flux model
    opt = Adam(learning_rate)
    loss(a,u,x) = Flux.Losses.mse(mod(a,x),u)

    function evalcb()
        n = size(atest,1)
        for i=1:n
            l = Flux.Losses.mse(mod(atest[:,i],itest[:,i]),utest[i])
            @show l
        end  
    end 


    for i=1:nepochs
        print("--------------------------------------\n")
        print("Epoch:$(i)\n")
        print("--------------------------------------\n")
        for n=1:size(atrain,1)
            Flux.train!(loss, Flux.params(mod), [(atrain[:,n],utrain[n],itrain[:,n])],opt,cb=evalcb)
        end 
        print("--------------------------------------\n")
    end 

    model
end 