abstract type TimeDependence end
struct TimeDependent <: TimeDependence end 
struct TimeIndependent <: TimeDependence end 

function complex_init(inputsize,outputsize)
    rand(ComplexF64,inputsize,outputsize)
end 

function complex_init(inputsize,outputsize,nmodes)
    rand(ComplexF64,inputsize,outputsize,nmodes)
end 

mutable struct ConvOp1D <: AbstractConvOperator
    nmodes::Int

    kernel::Array
    weight::Array

    inter::Array
    x̂::Array
    ŷ::Array

    function ConvOp1D(nmodes::Int, inputsizes::Vector)
        # input is expected to be of shape : 2 x nx x BS  
        s = inputsizes # inputsize x gridsize x batchsize
        
        W = complex_init(s[1],s[1])
        R = complex_init(s[1],s[1],ceil(Int,s[2]/2)+1)
        
        x̂ = zeros(s[1],s[2],ceil(Int,s[2]/2)+1)
        ŷ = similar()
        inter = zeros(ComplexF64,s[1],ceil(Int,s[2]/2)+1,s[3])
        new(nmodes,R,W,inter,x̂,ŷ)
    end 
end

Flux.@functor ConvOp1D

function (op::ConvOp1D)(x::AbstractArray)
    @unpack x̂,ŷ,inter,kernel,nmodes,weight = op
    x̂ = rfft(x)
    kernel[:,:,nmodes+1:end] .= zero(eltype(kernel)) 
    @tullio inter[o,g,bs] := kernel[o,i,g] * x̂[i,g,bs]
    inter = irfft(inter,size(x,2))
    @tullio ŷ[o,g,bs] := W[o,i] * x[i,g,bs]
    x̂ .+ ŷ
end 
 

mutable struct FNO1D <: FourierNeuralOperator
    a
    x
    t
    u
    
    input
    output
    
    DL
    NL
    nmodes

    function FNO1D(a::Array,x::Array,t::Array,u::Array,DL::Int,NL::Int,nmodes::Int)
        s = size(a)
        ng = s[1] 
        I = s[2]
        input = zeros(2,ng,I)
        output = zeros(1,ng,I)
        new(a,x,t,u,input,output,DL,NL,nmodes)
    end 

    function FNO1D(a::Vector{Array},x::Array,t::Array,DL::Int,NL::Int,nmodes::Int)
        s = size(a)
        ng = s[1] 
        I = s[2]
        input = zeros(length(a)+1,ng,I)
        output = zeros(1,ng,I)
        new(a,x,t,u,input,output,DL,NL,nmodes)
    end 
end

function (fno::FNO1D)()
    DL = fno.DL
    s = collect(size(fno.input))
    lifting = Dense(s[1]=>DL,gelu)
    s[1] = DL
    convlayers = [ConvOp1D(fno.nmodes, s) for _ in 1:fno.NL]
    reduction = Dense(DL=>size(fno.output)[1],gelu)
    Chain([lifting,convlayers...,reduction])
end 


function munge!(fno::FNO1D, ::TimeIndependent)
   input = fno.input 
   output = fno.output
   input[1,:,:] = fno.a[:,:]
   input[2,:,:] = fno.x[:,:]
   output[1,:,:] = permutedims(reshape(fno.u[:,1,:],(size(fno.u)[1],1,:)),(2,1,3))
   nothing
end 

function munge!(fno::FNO1D, ::TimeDependent)
    input = fno.input
    output = fno.output
    l = size(input,1)
    for j=1:l
        input[j,:,:] = fno.a[j][:,:]
    end 
    input[end,:,:] = fno.x[:,:]
    output[1,:,:] = permutedims(reshape(fno.u[:,1,:],(size(fno.u)[1],1,:)),(2,1,3))
    nothing 
end 

function split(input::Array,output::Array,ratio::Tuple=(0.7,0.2,0.1))
    I = size(input,3)
    
    k = floor(Int,ratio[1]*I)
    train_input = input[:,:,1:k]
    train_output = output[:,:,1:k]

    ks = k + 1
    kend = ks + floor(Int,ratio[2]*I)
    test_input = input[:,:,ks:kend]
    test_output = output[:,:,ks:kend]
    
    ks = kend + 1
    validation_input = input[:,:,ks:end]
    validation_output = output[:,:,ks:end]

    train_set = Dict("input"=>train_input,"output"=>train_output)
    test_set = Dict("input"=>test_input,"output"=>test_output)
    validation_set = Dict("input"=>validation_input,"output"=>validation_output)


    train_set,test_set,validation_set 
end

function learn(model::Chain,fno::FNO1D, learning_rate, nepochs, optimizer) 
    opt = optimizer(learning_rate)
    input = fno.input
    output = fno.output
    
    train_set,test_set,validation_set = split(input,output)
    loss(x,y) = Flux.Losses.mse(model(x),y)

    inputtest,outputtest = test_set["input"],test_set["output"]
    evalcb() = @show(loss(inputtest,outputtest))
    data = [(train_set["input"],train_set["output"])]
    Flux.@epochs nepochs Flux.train!(loss,Flux.params(model),data,opt,cb=evalcb)

    validation_set
end 