mutable struct DeepOpNet
    a
    x
    t
    u
    branch::Chain
    trunk::Chain

    input # (len(a),dims(x)+dims(t)) x bs 
    output # dims(u) x bs 

    function DeepOpNet(trunk, branch, raw_data)
        a,x,t,u = raw_data
        new(a,x,t,u,branch,trunk)
    end 
end 

function Base.show(io::IO,op::DeepOpNet)
    print(io,"DeepONet \n")
    print(io,"Trunk Net $(op.trunk)\n")
    print(io,"Branch Net $(op.branch)\n")
end 

Flux.@functor DeepOpNet

function (op::DeepOpNet)(input1::AbstractMatrix,input2::AbstractMatrix)
    inter1 = op.trunk(input1)
    inter2 = op.branch(input2)
    LinearAlgebra.diag(inter1' * inter2)
end 

function munge!(op::DeepOpNet,::OneD)
    st = size(op.t)
    sx = size(op.x)
    sa = size(op.a)
    I  = sx[2]
    input2 = zeros(2,st[1]*sx[1]*I)
    output = zeros(1,st[1]*sx[1]*I)
    input1 = zeros(sa[1],st[1]*sx[1]*I)

    o::Int = 1
    for j=1:I
        for i=1:st[1]
            for k = 1:sx[1]
                l = k + (i-1)*sx[1] + (j-1)*st[1]*sx[1]
                input2[1,l] = op.x[k]
                input2[2,l] = op.t[i]
                output[1,l] = op.u[k,i,j]
                input1[:,l] = op.a[:,j]
                @assert l==o
                o += 1
            end     
        end
    end

    # Shuffle data
    P = randperm(size(input1,2))
    input1 = input1[:,P]
    input2 = input2[:,P]
    output = output[:,P]
    
    op.input = [input1,input2]
    op.output = output
    nothing
end

function learn(op::DeepOpNet, ::OneD, nepochs = 10,learning_rate=0.01, ϵ=[0.7,0.2,0.1])

    ## Data splitting
    input = op.input
    output = op.output
    
    # Train, Test, Validate
    @assert sum(ϵ) ≈ 1.0 "Data Splitting ratio has to sum to 1.0"
    
    # Train
    ntrain = floor(Int,ϵ[1]*size(op.input[1],2))
    inputtrain = input[1][:,1:ntrain]
    intertrain = input[2][:,1:ntrain]
    outputtrain = output[:,1:ntrain]
    @show size(inputtrain)

    # Test
    ntest = floor(Int,ϵ[2]*size(op.input[1],2))
    inc = ntrain+1 : ntrain+1+ntest
    inputtest = input[1][:,inc]
    intertest = input[2][:,inc]
    outputtest = output[:,inc]
    @show size(inputtest)

    # Validate 
    inc = ntrain+ntest+1
    inputval = input[1][:,inc:end]
    interval = input[2][:,inc:end]
    outputval = output[:,inc:end]
    @show size(inputval)

    # Optimization

    opt = Flux.ADAM(learning_rate)
    loss(x,y,inter) = Flux.Losses.mse(op(x,inter),reshape(y,(:,)))
    evalcb() = @show(loss(inputtest,outputtest,intertest))
    data = [(inputtrain,outputtrain,intertrain)]
    Flux.@epochs nepochs Flux.train!(loss,Flux.params(op),data,opt,cb=evalcb)
    
    # Return a set for validation
    Dict("n"=>size(inputval,2),"input"=>inputval,"inter"=>interval,"output"=>outputval)
end