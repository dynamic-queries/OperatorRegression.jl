## 1D Automatic
using WaveSurrogates

begin
    dims = OneD()
    ninstances = 100
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
    write(sol,dims)
end 