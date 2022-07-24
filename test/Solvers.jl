using Plots

## 1D 
begin
    # Define grid, time parameters
    dims = OneD()
    ninstances = 2
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
    init = flatten(1,s,a,dims)
    sol = acousticWE(dims,method,init,s,t)

    # Test Ensemble Problem
    prob = EnsembleProblem(dims, method, a, s, t, ninstances)
    sol = solve(prob,acousticWE)
end 

## 2D 
begin 
    dims = TwoD()
    ninstances = 2
    nx = 50
    ny = 50
    nt = 100
    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 1.0
    tmin = 0.0
    tmax = 5.0
    x = xmin : (xmax-xmin)/nx : xmax
    y = ymin : (ymax-ymin)/ny : ymax
    t = tmin : (tmax-tmin)/nt : tmax

    kernel = SquaredExponential()
    s = (x,y)
    gp = GP(s,kernel)
    a = gp(ninstances,dims)
    A = reshape(Array(a,gp,TwoD()),(length(a),:))

    # Instance of the solver
    method = FiniteDiff()
    init = flatten(1,s,A,dims)
    sol = acousticWE(dims,method,init,s,t)


    # Ensemble problem
    prob = EnsembleProblem(dims,method,A,s,t,ninstances)
    sol = solve(prob,acousticWE)
end

## 2D Automatic 
begin
    dims = TwoD()
    ninstances = 2
    nx = 50
    ny = 50
    nt = 100
    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 1.0
    tmin = 0.0
    tmax = 5.0
    x = xmin : (xmax-xmin)/nx : xmax
    y = ymin : (ymax-ymin)/ny : ymax
    t = tmin : (tmax-tmin)/nt : tmax
    s = (x,y)
    method = FiniteDiff()

    prob = EnsembleProblem(dims,method,s,t,ninstances)
    sol = solve(prob,acousticWE)
end 

## 1D Automatic
begin
    dims = OneD()
    ninstances = 2
    nx = 1000
    nt = 100
    xmin = 0.0
    xmax = 1.0
    tmin = 0.0
    tmax = 1.0 
    x = xmin : (xmax-xmin)/nx : xmax
    t = tmin : (tmax-tmin)/nt : tmax
    method = FiniteDiff()
    s = x 

    prob = EnsembleProblem(dims,method,s,t,ninstances)
    sol = solve(prob,acousticWE)
end 

## 1D Save at start and end 
begin
    dims = OneD()
    ninstances = 2
    nx = 1000
    nt = 100
    xmin = 0.0
    xmax = 1.0
    tmin = 0.0
    tmax = 1.0 
    x = xmin : (xmax-xmin)/nx : xmax
    t = (tmin,tmax)
    method = FiniteDiff()
    s = x 

    prob = EnsembleProblem(dims,method,s,t,ninstances)
    sol = solve(prob,acousticWE)
end 
