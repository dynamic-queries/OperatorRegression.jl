using HDF5
using Flux

function redact(raw::Tuple,n::Int)
    a,x,t,u = raw
    a = a[:,1:n]
    x = x[:,1:n]
    t = t[:,1:n]
    u = u[:,:,1:n]
    a,x,t,u
end 

# Read data
filename = "data/End_1D_Wave_Equation"
file = h5open(filename)
raw_data = read(file["a"]),read(file["x"]),read(file["t"]),read(file["u"])
a,x,t,u = redact(raw_data,10)

# Conv Operator
nmodes = 14
inputsizes = (2,size(a,1),size(a,2))


# FNO Operator
NL = 4
DL = 32
nmodes = 14

lr = 0.01
nepochs = 500
optimizer = Flux.ADAM

fno = FNO1D(a,x,t,u,DL,NL,nmodes)
model = fno()
munge!(fno,TimeIndependent())

fno.input
fno.output

learn(model,fno,lr,nepochs,optimizer)
