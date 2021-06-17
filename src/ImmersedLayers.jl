#=
 Methods and types associated with single and double layer potentials
=#

module ImmersedLayers

using Reexport
using DocStringExtensions
#@reexport using CartesianGrids
#@reexport using RigidBodyTools
using CartesianGrids
using RigidBodyTools

using LinearAlgebra

using UnPack

export DoubleLayer, SingleLayer, MaskType, Mask, ComplementaryMask,
        DoubleLayer!, SingleLayer!, Mask!,
        SurfaceCache,
        regularize_normal!,normal_interpolate!,
        surface_curl!,surface_divergence!,surface_grad!,
        CLinvCT,GLinvD,nRTRn

abstract type LayerType{N} end

include("tools.jl")
include("cache.jl")
include("layers.jl")
include("surface_operators.jl")




end
