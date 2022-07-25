struct Random end 
struct TimeDependent end 
struct TimeIndependent end 
using Flux


# 1D Problem
function generate_data(::Random,::OneD)
    I = 20
    n = 20
    m = 30
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
    inputsize = size(a,1)
    outputsize = size(u)[2],size(u)[3] 
    npoints, inputsize, outputsize 
end 

dims = OneD()
kind = Random()
raw_data = generate_data(kind,dims)
npoints, inputsize, outputsize = metadata(raw_data,dims)


struct FNOCache
    input
    output

    function FNOCache(metadata,::OneD,::TimeInDependent)

    end 

    function FNOCache(metadata,::TwoD,::TimeIndependent)

    end 

    function FNOCache(metadata,::OneD,::TimeDependent)

    end 

    function FNOCache(metadata,::TwoD,::TimeDependent)
        
    end 
end 


struct FNO 
    a
    x
    t
    u 
    dims
    timedependence
    cache::FNOCache
    nconv::Int
    nmodes::Int
    wlifting_layer::Int
    model::Flux.Chain

    function FNO(a,x,t,u,dims,tdepend)
        nconv = 4
        nmodes = 14 
        wlifting_layer = 1024
        cache = FNOCache(metadata((a,x,t,u)),dims,tdepend)

        if typeof(dims) == OneD
            conv = Conv1D(nmodes,wlifting_layer,wlifting_layer)
        elseif typeof(dims) == TwoD
            conv = Conv2D(nmodes,wlifting_layer,wlifting_layer)
        elseif typeof(dims) == ThreeD
            conv = Conv3D(nmodes,wlifting_layer,wlifting_layer)
        end 

        model = ...

        new(a,x,t,u,dims,tdepend,cache,nconv,nmodes,wlifting_layer)
    end 

    function FNO(a,x,t,u,dims,tdepend,nconv,nmodes,wlifting_layer)
        cache = FNOCache(metadata((a,x,t,u)),dims,tdepend)
        new(a,x,t,u,dims,tdepend,cache,nconv,nmodes,wlifting_layer)
    end     
end 


function munge!(fno::FNO,::OneD,::TimeIndependent)

end 

Flux.@functor FNO

function (fno::FNO)(input::Array)
    
end 



# 1D Problem, TimeIndependent
dims = OneD()
kind = TimeIndependent()
a,x,t,u = generate_data(Random(),OneD())
nfl = 4
nmodes = 14 
wlifting_layer = 1024
model = FNO(a,x,t,u,dims,kind)
munge!(model)