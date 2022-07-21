mutable struct EnsembleProblem 
    dims::Dimension
    method::Method
    u₀::Tuple # Contains vectorized values and locations of the boundaries in the vector.
    S::Tuple
    t::Union{StepRange,StepRangeLen}

    function EnsembleProblem(dims,method,u0,S,t) 
        new(dims,method,u0,S,t)
    end
    
    function EnsembleProblem(dims,u0,S,t)
        new(dims,FiniteDiff(),u0,S,t)
    end 
end


mutable struct EnsembleSolution 
    prob::EnsembleProblem
    solver
    ninstances::Int
    solutions::Vector

    function EnsembleSolution(prob,solver,ninst)
        new(prob,solver,ninst)
    end 
end 


function flatten(a::Array,::OneD)

end

function flatten(a::Array,::TwoD)

end 

function flatten(a::Array,::ThreeD)

end 

function solve(prob::EnsembleProblem,solver)
    sol = EnsembleSolution(prob,solver,size(prob.u₀,2))
    n = sol.ninstances
    for i = 1:n
        push!(sol.solutions, solver(prob.S,flatten(prob.u₀,prob.dims),prob.t))
    end     
    sol
end 


function munge(sol::EnsembleSolution)

end 
#=============================================================#