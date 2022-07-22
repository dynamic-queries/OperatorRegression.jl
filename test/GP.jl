begin
    # 1D 
    # Define grid, time parameters
    dims = OneD()
    ninstances = 100
    nx = 100
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
end 

begin
    ## 2D problem 
    dims = TwoD()
    ninstances = 10
    nx = 500
    ny = 500
    nt = 10
    xmin = 0.0
    xmax = 1.0 
    ymin = 0.0 
    ymax = 1.0 
    tmin = 0.0 
    tmax = 1.0 
    x = xmin : (xmax - xmin)/nx : xmax
    y = ymin : (ymax - ymin)/ny : ymax 
    t = tmin : (tmax - tmin)/nt : tmax 
    grid = Grid(x,y)
    state = flatten(grid,dims)

    # Generate intial conditions
    kernel = SquaredExponential()
    gp = GP(state,kernel)
    a = gp(ninstances,dims)
    A = Array(a,gp,TwoD())
    heatmap(A,gp)
end