#=
 Methods and types associated with single and double layer potentials
=#

module ImmersedLayers

using Reexport
@reexport using CartesianGrids
@reexport using RigidBodyTools

using LinearAlgebra

export DoubleLayer, SingleLayer, MaskType, Mask, ComplementaryMask

abstract type LayerType{N} end

include("tools.jl")
include("layers.jl")



end
