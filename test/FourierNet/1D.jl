using HDF5

# Read data
filename = "data/End_1D_Wave_Equation"
file = h5open(filename)
a,x,t,u = read(file["a"]),read(file["x"]),read(file["t"]),read(file["u"])
raw_data = (a,x,t,u)

# Conv Operator

size(a) 

size(x) 

size(t) 

size(u)


# FNO Operator