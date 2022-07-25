#=============================================================#
abstract type Dimension end
struct OneD <: Dimension end
struct TwoD <: Dimension end  
struct ThreeD <: Dimension end 

Base.Int(::OneD) = Int(1)
Base.Int(::TwoD) = Int(2)
Base.Int(::ThreeD) = Int(3)

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
    S::Union{Tuple,StepRangeLen}
    kernel::Kernel

    function GP(state,kernel::Kernel)
        new(state,kernel)
    end 

    function GP(state)
        new(state,SquaredExponential())
    end 

end 

function (gp::GP)(ninstances::Int,::OneD) 
    x = gp.S
    n = length(x) 
    m = zeros(n)
    K = zeros(n,n)
    σ = 0.5

    for i=1:n 
        for j=1:n 
            K[i,j] = exp(-(x[i]-x[j])^2/σ^2)
        end
    end 
    K += 1e-6*I(n)
    dist = MvNormal(m,K)
    rand(dist,ninstances)
end 

function (gp::GP)(ninstances::Int,::TwoD)
    x = gp.S[1]
    y = gp.S[2] 
    nx = length(x)
    ny = length(y)
    n = nx*ny
    meanx = zeros(nx)
    meany = zeros(ny)
    σ = 0.5 
    Kx = zeros(nx,nx)
    Ky = zeros(ny,ny)
    for i=1:nx 
        for j=1:nx
            Kx[i,j] = exp(-(x[i]-x[j])^2/σ^2)
        end 
    end 

    for i=1:ny 
        for j=1:ny
            Ky[i,j] = exp(-(y[i]-y[j])^2/σ^2)
        end 
    end 

    Kx += 1e-6*I(nx)
    Ky += 1e-6*I(ny)
    dist = MvNormal(meanx,Kx)
    gaussx = rand(dist,ninstances)
    dist = MvNormal(meany,Ky)
    gaussy = rand(dist,ninstances)
    gaussx,gaussy
end 

function Base.Array(a::Tuple,gp::GP,::TwoD)
    ninstances = size(a[1],2)
    nx = length(gp.S[1])
    ny = length(gp.S[2])
    A = Array{Float32,3}(undef,ninstances,nx,ny)
    for i=1:ninstances
        for j=1:nx
            for k=1:ny
                A[i,j,k] = a[1][j,i]*a[2][k,i]
            end 
        end     
    end
    A 
end 


function Plots.plot(A::AbstractMatrix,gp::GP)
    ninstances = size(A,2)
    anim = @animate for i=1:ninstances
        plot(A[:,i],ylim=[-2.0,2.0])
    end 
    gif(anim,"gp.gif",fps=10)
end 

function Plots.heatmap(A::AbstractArray,gp::GP) 
    ninstances = size(A,1)
    anim = @animate for i=1:ninstances
        heatmap(A[i,:,:],clim=(-1.0,1.0))
    end
    gif(anim,"2d-gp.gif",fps=10)
end 

#=============================================================#

