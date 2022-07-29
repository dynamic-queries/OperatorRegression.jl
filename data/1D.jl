## 1D Automatic
using WaveSurrogates
using HDF5

function truncate(sol,::OneD,filename="1DJumpStep")
    a,x,t,u = Array(sol,dims)
    tnew = reshape(t[end,:],(1,1))
    unew = reshape(u[:,end,:],(size(u,1),1,:))
        
    file = h5open(filename,"w")
    file["a"] = a
    file["x"] = x
    file["t"] = tnew
    file["u"] = unew
    close(file)   
end 


begin # Time series
    dims = OneD()
    ninstances = 1
    nx = 1000
    nt = 10
    xmin = 0.0
    xmax = 1.0
    tmin = 0.0
    tmax = 5.0 
    x = xmin : (xmax-xmin)/nx : xmax
    t = tmin : (tmax-tmin)/nt : tmax
    method = FiniteDiff()
    s = x 

    prob = EnsembleProblem(dims,method,s,t,ninstances)
    sol = solve(prob,acousticWE)
    write(sol,dims)
end 

# Save the last time step alone for ease of testing the FNO
truncate(sol,dims)
