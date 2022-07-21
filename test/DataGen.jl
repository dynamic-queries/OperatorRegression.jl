## 1D problem
# Define grid, time parameters
ninstances = 10
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
a = GP(x,kernel,ninstances)

# Package this into a struct
method = FiniteDiff()
dims = OneD()
solver = wave_problem()
problem = EnsembleProblem(dims,method,a,x,t)

# Send struct to the solver 
solution = solve(problem,solver) # Returns an ensemble solution

# Munge the solution to the form a,inter,u 
a,inter,u = munge(solution)


## 2D problem 
ninstances = 10
nx = 20 
ny = 30
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
state = Array(grid)

# Generate intial conditions
kernel = SquaredExponential()
a = GP(state,kernel,ninstances)

# Package into a struct and send it to the solver
method = FiniteDiff()
dims = TwoD()
solver = wave_problem()
problem = EnsembleProblem(dims,method,a,state,t)
solution = solve(problem,solver)

# Munge the data and send it to DeepOpNet
a,inter,u = munge(solution)


## 3D problem 
ninstances = 10
nx = 10
ny = 12
nz = 14
nt = 10
xmin = 0.0 
xmax = 1.0 
ymin = 0.0
ymax = 1.0 
zmin = 0.0 
zmax = 1.0 
tmin = 0.0 
tmax = 1.0 
x = xmin : (xmax - xmin)/nx : xmax
y = ymin : (ymax - ymin)/ny : ymax 
z = zmin ; (zmax - zmin)/nz : zmax
t = tmin : (tmax - tmin)/nt : tmax
grid = Grid(x,y,z)
state = Array(grid)

# Get the initial conditions
kernel = SquaredExponential()
a = GP(state,kernel,ninstances)

# Package this into a solver
method = FiniteDiff()
dims = ThreeD()
solver = wave_problem()
problem = EnsembleProblem(dims,method,a,state,t)
solution = solve(problem,solver)

# Munge the solution 
a,inter,u = munge(solution)