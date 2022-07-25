using WaveSurrogates

begin
    dims = TwoD()
    ninstances = 100
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
    write(sol,dims)
end 