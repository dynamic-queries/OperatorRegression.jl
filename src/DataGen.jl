#=============================================================#
abstract type Dimension end
struct OneD <: Dimension end
struct TwoD <: Dimension end  
struct ThreeD <: Dimension end 
#=============================================================#

#=============================================================#
abstract type Method end 
struct FiniteDiff <: Method end 
struct PseudoSpectral <: Method end 
struct Spectral <: Method end 
#=============================================================#

#=============================================================#

abstract type Kernel end 
struct SquaredExponential <: Kernel end 

#=============================================================#

mutable struct Grid
    x::Union{StepRange,StepRangeLen}
    y::Union{StepRange,StepRangeLen}
    z::Union{StepRange,StepRangeLen}

    function Grid(x::Union{StepRange,StepRangeLen})
        new(x)
    end 

    function Grid(x,y)
        new(x,y)
    end 

    function Grid(x,y,z) 
        new(x,y,z)
    end 
end 

function flatten(grid::Grid,dim::Dimension)
    s = nothing
    if dim == OneD()
        s = (grid.x)
    elseif dim == TwoD()
        s = (grid.x, grid.y)
    elseif dim == ThreeD() 
        s = (grid.x, grid.y, grid.z)
    end 
    s
end 

#=============================================================# 

mutable struct GP
    S::Tuple
    kernel::Kernel

    function GP(state::Array,kernel::Kernel)
        new(state,kernel)
    end 

    function GP(state::Array)
        new(state,SquaredExponential())
    end 

end 


function (gp::GP)(ninstances::Int,OneD()) 
    x = gp.S[1] 
    n = length(x) 
    m = zeros(n)
    K = zeros(n,n)
    σ = 0.5

    for i=1:n 
        for j=1:n 
            K[i,j] = exp(-(x[i]-x[j]/σ)^2)
        end
    end 

    dist = MvNormal(m,K)
    rand(dist,ninstances)
end 


function (gp::GP)(ninstances::Int,TwoD())
    x = gp.S[1]
    y = gp.S[2] 
    nx = length(x)
    ny = length(y)
    n = nx*ny
    mean = zeros(nx,ny)
    σ = 0.5 
    K = zeros(n,n)
    for i=1:n 
        for j=1:n
            K[i,j] = exp(-(norm([x[i] y[i]]-[x[j] y[j]])/σ)^2)
        end 
    end 
    dist = MvNormal(mean,K)
    rand(dist,ninstances)
end 

function (gp::GP)(ninstances::Int,ThreeD())
    x = gp.S[1]
    y = gp.S[2] 
    z = gp.S[3]
    nx = length(x)
    ny = length(y)
    nz = length(z) 
    mean = zeros(nx,ny,nz)
    n = nx*ny*nz
    K = zeros(n,n)
    for i=1:n
        for j=1:n
            K[i,j] = exp(-(norm([x[i] y[i] z[i]]-[x[j] y[j] z[j]])/σ)^2)
        end 
    end 
    dist = MvNormal(mean,K)
    rand(dist,ninstances)
end 


#=============================================================#

