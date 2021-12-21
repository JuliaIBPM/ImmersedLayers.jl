#=
 Methods and types associated with single and double layer potentials
=#

module ImmersedLayers

using Reexport
using DocStringExtensions
@reexport using CartesianGrids
@reexport using RigidBodyTools
@reexport using ConstrainedSystems

import CartesianGrids: Laplacian

#using CartesianGrids
#using RigidBodyTools

using LinearAlgebra

using UnPack

export DoubleLayer, SingleLayer, MaskType, Mask, ComplementaryMask,
        DoubleLayer!, SingleLayer!, Mask!,
        SurfaceCache,SurfaceScalarCache,SurfaceVectorCache,
        @ilmproblem,construct_system,update_system,update_system!,
        BasicScalarILMProblem,BasicVectorILMProblem,prob_cache,
        AbstractScalingType,GridScaling,IndexScaling,
        BasicILMCache,ILMSystem,
        ODEFunctionList,zeros_sol,
        similar_grid,similar_gridgrad,similar_gridcurl,similar_surface,
        zeros_grid,zeros_gridgrad,zeros_gridcurl,zeros_surface,
        ones_grid,ones_gridgrad,ones_gridcurl,ones_surface,
        x_grid,y_grid,x_gridcurl,y_gridcurl,
        AbstractExtraILMCache,AbstractScalarILMProblem,AbstractVectorILMProblem,
        ConvectiveDerivativeCache,convective_derivative, convective_derivative!,
        regularize!, interpolate!,
        regularize_normal!,normal_interpolate!,
        regularize_normal_cross!,normal_cross_interpolate!,
        surface_curl!,surface_divergence!,surface_grad!,inverse_laplacian!,
        laplacian!,Laplacian,
        surface_curl_cross!,surface_divergence_cross!,surface_grad_cross!,
        mask,mask!,complementary_mask,complementary_mask!,
        create_CLinvCT,create_CL2invCT,create_GLinvD,create_nRTRn,create_RTLinvR,
        create_GLinvD_cross,create_surface_filter,
        solve

abstract type LayerType{N} end

"""
$(TYPEDEF)

A system of operators and caches for immersed layer problems. This is constructed
by [`__init`](@ref)
"""
mutable struct ILMSystem{static,PT,PHT,BCF,FF,DTF,MTF,BCT,ECT}

  phys_params :: PHT
  bc :: BCF
  forcing :: FF
  timestep_func :: DTF
  motions :: MTF
  base_cache :: BCT
  extra_cache :: ECT

end

include("cartesian_extensions.jl")
include("tools.jl")
include("cache.jl")
include("problem.jl")
include("system.jl")
include("layers.jl")
include("grid_operators.jl")
include("surface_operators.jl")
include("matrix_operators.jl")
include("timemarching.jl")

include("plot_recipes.jl")

@deprecate DoubleLayer surface_divergence!
@deprecate SingleLayer regularize!


end
