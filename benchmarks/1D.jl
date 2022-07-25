using BenchmarkTools
using Profile

"""
    One instance.
    One dimensional problem.
    n = 1000 , grid points
    m = 100, time steps
"""
function instance1() 
    dims = OneD()
    ninstances = 1
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
    sol
end 

@benchmark solution = instance1()

"""
BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):    4.589 s … 248.826 s  ┊ GC (min … max): 0.00% … 1.77%
 Time  (median):     126.708 s              ┊ GC (median):    1.74%
 Time  (mean ± σ):   126.708 s ± 172.702 s  ┊ GC (mean ± σ):  1.74% ± 1.25%

  █                                                         █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  4.59 s          Histogram: frequency by time          249 s <

 Memory estimate: 390.66 MiB, allocs estimate: 12882698.
"""

"""
    1 instance.
    One dimensional problem.
    n = 1000 , grid points
    m = 2, time steps
"""
function instance2() 
    dims = OneD()
    ninstances = 1
    nx = 1000
    nt = 1
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
    sol
end

@profile solution = instance2()

# For one instance
"""
BenchmarkTools.Trial: 2 samples with 1 evaluation.
 Range (min … max):  4.447 s …   4.474 s  ┊ GC (min … max): 0.79% … 1.97%
 Time  (median):     4.461 s              ┊ GC (median):    1.38%
 Time  (mean ± σ):   4.461 s ± 19.184 ms  ┊ GC (mean ± σ):  1.38% ± 0.84%

  █                                                       █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  4.45 s         Histogram: frequency by time        4.47 s <

 Memory estimate: 383.19 MiB, allocs estimate: 12390543.

"""