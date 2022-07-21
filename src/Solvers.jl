"""
    acousticWE : δₜₜu = c² δₓₓu + f

Solves the acoustic wave equation in One dimension.
Assumes homogenous dirichlet boundary condition.
"""
function acousticWE(dims::OneD(),
                       method::FiniteDiff(),
                       a::Tuple,
                       s::Tuple,
                       t::Union{StepRange,StepRangeLen}
                       )
    function wave!(du,u,p,t)
        c = p[2]
        dx2 = c/p[1]^2
        for i=2:length(u)-1
            du[i] = dx2*(u[i+1]+u[i-1]-2*u[i])
        end 
    end 
    
    # Set Boundary conditions
    u₀,blocs = a
    for i=1:length(blocs)
        u₀[blocs[i]] = 0.0
    end 

    # Setup ODE problem
    c = 1.0
    tspan = (t[1],t[end])
    x = s[1] 
    dx = x[2]-x[1]
    p = [dx,c]
    prob = ODEProblem(wave!,a,tspan,p)
    sol = OrdinaryDiffEq.solve(prob,TRBDF2(),saveat=t)
    sol
end 

""" 
    acousticWE : δₜₜu = c² Δu + f

Solves the acoustic Wave Equation in two dimensions. 
Assumues homogenous dirichlet boundary conditions.
"""
function acousticWE(dims::TwoD(),
    method::FiniteDiff(),
    a::Tuple,
    s::Tuple,
    t::Union{StepRange,StepRangeLen}
    )

    # Forcing function 
    function wave!(du,u,p,t)
        dx2 = 1/p[1]^2
        dy2 = 1/p[2]^2
        cx = p[3]
        cy = p[4] 
        nx = p[5]
        ny = p[6]
        n = nx*ny

        for i=2:nx-1
            for j=2:ny-1
                k = j + (i-1)*ny
                k̂ = n + k
                du[k] = u[k̂]
                du[k̂] = cx*dx2*(u[k+1]+u[k-1]-2*u[k]) + cy*dy2*(u[k+ny]+u[k-ny]-2*u[k])
            end   
        end 
    end 

    # Boundary conditions
    u₀, blocs = a
    for i=1:length(blocs)
        u₀[blocs[i]] = 0.0
    end 
    
    # Setup the ODE problem
    x = s[1]
    y = s[2]
    tspan = (t[1],t[end])
    dx = x[2]-x[1]
    dy = y[2]-y[1]
    cx = 1.0
    cy = 1.0
    nx = length(x) 
    ny = length(y) 
    p = [dx,dy,cx,cy,nx,ny]
    prob = ODEProblem(wave!,u₀,tspan,p)
    sol = OrdinaryDiffEq.solve(prob,TRBDF2(),saveat=t)
    sol
end 