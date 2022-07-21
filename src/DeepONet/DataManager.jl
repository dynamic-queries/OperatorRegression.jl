
# a generated from a GP

# prob

# solution, containing u,t


nx = 2
ny = 3
nt = 5
a = rand(nx,ny)
u = rand(nt,nx,ny)
t = rand(nt)



# Generate Random data for testing the Constructor

# Number of dimnesions of the grid
ndims = 1
# Number of instances of the input a 
na = 2
# Number of grid points 
res = 2 
m = (2<<res)^ndims
# Number of time points 
nt = 2<<(res+1)
# Total number of data points 
ND = na * m * nt

## Munged Data
# Input function a
a = repeat(rand(m)',ND) # Each row is a data point
# Output function 
u = rand(ND)
# Cogs
inter = rand(ND,ndims+1) # One for x and the other for t )
