mutable struct EnsembleProblem 
    dims::Dimension
    method::Method
    ninstances::Int
    u₀::Array
    S::Union{Tuple,StepRangeLen}
    t::Union{StepRange,StepRangeLen,Tuple}

    function EnsembleProblem(dims::Dimension,
                            method::Method,
                            u0::Array,
                            S::Union{Tuple,StepRangeLen},
                            t::Union{StepRange,StepRangeLen,Tuple},
                            ninstances::Int) 
        new(dims,method,ninstances,u0,S,t)
    end
    
    function EnsembleProblem(dims::Dimension,
                            u0::Array,
                            S::Union{Tuple,StepRangeLen},
                            t::Union{StepRange,StepRangeLen,Tuple},
                            ninstances::Int)
        new(dims,FiniteDiff(),ninstances,u0,S,t)
    end 

    function EnsembleProblem(dims::Dimension,
                            method::Method,
                            S::Union{Tuple,StepRangeLen},
                            t::Union{StepRange,StepRangeLen,Tuple},
                            ninstances::Int)
        kernel = SquaredExponential()
        gp = GP(S,kernel)
        u0 = gp(ninstances,dims)
        if typeof(dims) == TwoD
            u0 = reshape(Array(u0,gp,dims,),(length(u0),:))
        end 
        new(dims,method,ninstances,u0,S,t)
    end 

    function EnsembleProblem(dims::Dimension,
                            S::Union{Tuple,StepRangeLen},
                            t::Union{StepRange,StepRangeLen,Tuple},
                            ninstances::Int)
        kernel = SquaredExponential()
        gp = GP(S,kernel)
        u0 = gp(ninstances,dims)
        if typeof(dims) == TwoD
        u0 = reshape(Array(u0,gp,dims,),(length(u0),:))
        end 
        new(dims,FiniteDiff(),ninstances,u0,S,t)
    end 
end

mutable struct EnsembleSolution 
    prob::EnsembleProblem
    solver
    ninstances::Int
    solutions::Vector

    function EnsembleSolution(prob,solver,ninst)
        new(prob,solver,ninst,[])
    end 
end 

function flatten(i::Int,s::Union{Tuple,StepRangeLen},a::Union{Array,StepRangeLen},::OneD)
    n = length(s)
    v = vcat(a[:,i],zeros(n))
    bclocs = [1,n,n+1,2*n]
    (v,bclocs)
end 

function flatten(i::Int,s::Tuple,a::Union{Array,StepRangeLen},::TwoD)
    nx = length(s[1])
    ny = length(s[2])
    n = nx*ny
    @assert n == size(a,2)
    v = vcat(a[i,:],zeros(n))
    bclocs = vcat(collect(Int,1:nx),collect(Int,(ny-1)+1:(ny-1)+nx),collect(Int,1:ny:n),collect(Int,ny:ny:n))
    (v,bclocs)
end 

function solve(prob::EnsembleProblem,solver)
    sol = EnsembleSolution(prob,solver,prob.ninstances)
    n = prob.ninstances
    for i = 1:n
        push!(sol.solutions, solver(prob.dims,prob.method,flatten(i,prob.S,prob.u₀,prob.dims),prob.S,prob.t))
    end     
    sol
end 

function Base.Array(sol::EnsembleSolution)

end 

function reduce(sol::EnsembleSolution,nmodes::Int)

end 

function write(sol::EnsembleSolution)

end 
#=============================================================#