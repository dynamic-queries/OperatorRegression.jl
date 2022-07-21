using LinearAlgebra
using Plots

abstract type Dimension end
struct OneD <: Dimension end
struct TwoD <: Dimension end  
struct ThreeD <: Dimension end 

abstract type Method end 
struct FiniteDiff <: Method end 
struct PseudoSpectral <: Method end 
struct Spectral <: Method end 