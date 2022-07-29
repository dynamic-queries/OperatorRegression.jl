using HDF5

# Read data
filename = "data/End_1D_Wave_Equation"
file = h5open(filename)
a,x,t,u = read(file["a"]),read(file["x"]),read(file["t"]),read(file["u"])
raw_data = (a,x,t,u)

# Conv Operator
nmodes = 14
inputsizes = (2,size(a,1),size(a,2))


# FNO Operator