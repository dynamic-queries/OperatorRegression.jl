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
            u0 = reshape(Array(u0,gp,dims,),(size(u0[1],2),:))
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
        u0 = reshape(Array(u0,gp,dims,),(size(u0[1],2),:))
        end 
        new(dims,FiniteDiff(),ninstances,u0,S,t)
    end 
end

function Base.show(io::IO, prob::EnsembleProblem)
    print("Ensemble Problem \n")
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

function Base.show(io::IO, sol::EnsembleSolution)
    print("Ensemble Solution \n")
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
    p = Progress(n,1,"Solving Ensemble Problem.",50)
    for i = 1:n
        push!(sol.solutions, solver(prob.dims,prob.method,flatten(i,prob.S,prob.u₀,prob.dims),prob.S,prob.t))
        next!(p)
    end     
    sol
end 

function Base.Array(sol::EnsembleSolution, ::OneD)
    I = sol.ninstances
    n = length(sol.prob.S)
    m = length(sol.prob.t)

    a = zeros(n,I)
    x = zeros(n,I)
    t = zeros(m,I)
    u = zeros(n,m,I)

    for i=1:I
        a[:,i] = sol.solutions[i].prob.u0[1:Int(end/2)]
        x[:,i] = collect(sol.prob.S)
        t[:,i] = collect(sol.prob.t)
        u[:,:,i] = Array(sol.solutions[i])[1:Int(end/2),:]
    end 

    (a,x,t,u)
end 

function Base.Array(x::StepRangeLen,y::StepRangeLen)
    nx = length(x)
    ny = length(y) 
    X = zeros(nx,ny)
    Y = zeros(nx,ny)
    for i=1:nx
        for j=1:ny
            X[i,j] = x[i]
            Y[i,j] = y[j]
        end 
    end 
    (X,Y)
end 

function Base.Array(sol::EnsembleSolution, ::TwoD)
    I = sol.ninstances
    nx = length(sol.prob.S[1])
    ny = length(sol.prob.S[2])
    m = length(sol.prob.t)
    
    a = zeros(nx,ny,I)
    x = zeros(nx,ny,I)
    y = zeros(nx,ny,I)
    t = zeros(m,I)
    u = zeros(nx,ny,m,I)
    
    x1,y1 = Array(sol.prob.S[1],sol.prob.S[2])
    
    for i=1:I
        a[:,:,I] = reshape(sol.solutions[i].prob.u0[1:Int(end/2)],(nx,ny))
        x[:,:,I] = x1
        y[:,:,I] = y1
        t[:,I] = collect(sol.prob.t)
        u[:,:,:,I] = reshape(Array(sol.solutions[i])[1:Int(end/2),:],(nx,ny,:))
    end 

    (a,(x,y),t,u)
end 


function Base.write(sol::EnsembleSolution,::OneD,fname="1D_Wave_Equation")
    a,x,t,u = Array(sol,OneD())
    file = h5open(fname,"w")
    file["a"] = a
    file["x"] = x
    file["t"] = t
    file["u"] = u
    close(file)
end 

function Base.write(sol::EnsembleSolution,::TwoD,fname = "2D_Wave_Equation")
    a,(x,y),t,u = Array(sol,TwoD())
    file = h5open(fname,"w")
    file["a"] = a
    file["x"] = x
    file["y"] = y
    file["t"] = t
    file["u"] = u
    close(file)    
end 
#=============================================================#