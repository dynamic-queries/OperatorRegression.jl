function complex_init(inputsize,outputsize)

end 

function complex_init(inputsize,outputsize,nmodes)

end 

mutable struct ConvOp1D <: AbstractConvOperator
    v::Array
    nmodes::Int

    kernel::Array
    weight::Array

    inter::Array
    x̂::Array
    ŷ::Array

    function ConvOp1D(nmodes::Int, inputsizes::Tuple)
        # input is expected to be of shape : 2 x nx x BS  
        s = inputsizes # inputsize x gridsize x batchsize
        
        W = complex_init(s[1],s[1])
        R = complex_init(s[1],s[1],ceil(Int,s[2]/2)+1)
        
        x̂ = similar(input)
        ŷ = similar(input)
        inter = zeros(ComplexF64,s[1],ceil(Int,s[2]/2)+1,s[3])
        new(input,nmodes,R,W,inter,x̂,ŷ)
    end 
end

Flux.@functor ConvOp1D

function (op::ConvOp1D)(x::AbstractArray)
    @unpack x̂,ŷ,inter,kernel,nmodes,weight = op
    x̂ .= rfft(x)
    kernel[:,nmodes+1:end,:] .= zero(eltype(kernel)) 
    @tullio inter[o,g,bs] := kernel[o,i,g] * x̂[i,g,bs]
    inter .= irfft(inter,size(x,2))
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
    
    function FNO1D(a,x,t,u,input,output,nmodes)
        new(a,x,t,u,input,output,nmodes)
    end 
end

function model(fno::FNO1D)
    DL = fno.DL
    s = size(fno.input)
    lifting = Dense(s[1]=>DL,init=complex_init(DL,s[1]),gelu)
    s[1] = DL
    convlayers = [ConvOp1D(fno.nmodes, s) 1:fno.NL]
    reduction = Dense(DL=>size(fno.output)[1],init=complex_init(DL,size(output[1])),gelu)
    Chain(lifting,convlayers...,reduction)
end 

function munge!(fno::FNO1D)
   @unpack a,x,t,u = fno
   @assert size(a,1) == size(x,1)
   I = size(a,2)
   ng = size(a,1)
   input = zeros(2,ng,I)
   output = zeros(1,ng,I)
end 

function learn(fno::FNO1D) 

end 