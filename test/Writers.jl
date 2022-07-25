begin 
# Define grid, time parameters
    dims = OneD()
    ninstances = 5
    nx = 1000
    nt = 100
    xmin = 0.0
    xmax = 1.0
    tmin = 0.0
    tmax = 1.0 
    x = xmin : (xmax-xmin)/nx : xmax
    t = tmin : (tmax-tmin)/nt : tmax

    # Generate initial conditions
    kernel = SquaredExponential()
    s = x
    gp = GP(s,kernel)
    a = gp(ninstances,dims)
    plot(a,gp)

    # Test instance of the solver
    method = FiniteDiff()

    # Test Ensemble Problem
    prob = EnsembleProblem(dims, method, a, s, t, ninstances)
    sol = solve(prob,acousticWE)

    # Test Array(sol)
    raw_data = Array(sol,dims)

    # Write Array
    using WaveSurrogates:write
    using HDF5
    write(sol,dims)

    # Read back the data
    filename = "/home/dynamic-queries/.julia/dev/WaveSurrogates.jl/1D_Wave_Equation"
    file = h5open(filename,"r")
    a = read(file["a"])
    u = read(file["u"])
    t = read(file["t"])
    x = read(file["x"])
    close(file)
end 


## 2D Automatic 
begin
    dims = TwoD()
    ninstances = 2
    nx = 20
    ny = 20
    nt = 10
    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 1.0
    tmin = 0.0
    tmax = 2.0
    x = xmin : (xmax-xmin)/nx : xmax
    y = ymin : (ymax-ymin)/ny : ymax
    t = tmin : (tmax-tmin)/nt : tmax
    s = (x,y)
    method = FiniteDiff()

    prob = EnsembleProblem(dims,method,s,t,ninstances)
    sol = solve(prob,acousticWE)

    # Test Array(x,y)
    x,y = Array(prob.S[1],prob.S[2])

    using Plots
    heatmap(x)
    heatmap(y)

    # Test Array(sol)
    a,x,t,u = Array(sol,dims)

    # Write solutions
    using WaveSurrogates:write
    write(sol,dims)

    # Read Solutions back
    filename = "/home/dynamic-queries/.julia/dev/WaveSurrogates.jl/2D_Wave_Equation"
    file = h5open(filename,"r")
    a = read(file["a"])
    u = read(file["u"])
    t = read(file["t"])
    x = read(file["x"])
    y = read(file["y"])
    close(file)
end 