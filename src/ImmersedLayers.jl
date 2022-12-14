#=
 Methods and types associated with single and double layer potentials
=#

module ImmersedLayers

using Reexport
using DocStringExtensions
@reexport using CartesianGrids
@reexport using RigidBodyTools
@reexport using ConstrainedSystems
#@reexport using SpaceTimeFields

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
        ODEFunctionList,zeros_sol,init_sol,
        get_grid,get_bodies,
        similar_grid,similar_gridgrad,similar_gridcurl,similar_griddiv,similar_gridgradcurl,
        similar_surface,similar_surfacescalar,
        zeros_grid,zeros_gridgrad,zeros_gridcurl,zeros_griddiv,zeros_gridgradcurl,
        zeros_surface,zeros_surfacescalar,
        ones_grid,ones_gridgrad,ones_gridcurl,ones_griddiv,ones_gridgradcurl,
        ones_surface,ones_surfacescalar,
        x_grid,y_grid,x_gridcurl,y_gridcurl,x_griddiv,y_griddiv,x_gridgrad,y_gridgrad,
        AbstractExtraILMCache,AbstractScalarILMProblem,AbstractVectorILMProblem,
        ConvectiveDerivativeCache,convective_derivative, convective_derivative!,
        RotConvectiveDerivativeCache,w_cross_v!,w_cross_v,
        regularize!, interpolate!,
        regularize_normal!,normal_interpolate!,
        regularize_normal_symm!,normal_interpolate_symm!,
        regularize_normal_cross!,normal_cross_interpolate!,
        regularize_normal_dot!,normal_dot_interpolate!,
        surface_curl!,surface_divergence!,surface_grad!,inverse_laplacian!,
        laplacian!,Laplacian,
        surface_curl_cross!,surface_divergence_cross!,surface_grad_cross!,
        surface_divergence_symm!,surface_grad_symm!,
        mask,mask!,complementary_mask,complementary_mask!,
        ScalarPotentialCache,VectorPotentialCache,VectorFieldCache,
        vectorpotential_from_masked_curlv!,scalarpotential_from_masked_divv!,
        vectorpotential_from_curlv!,vecfield_from_vectorpotential!,masked_curlv_from_curlv_masked!,
        scalarpotential_from_divv!,masked_divv_from_divv_masked!,vecfield_from_scalarpotential!,
        divv_masked_from_masked_divv!,curlv_masked_from_masked_curlv!,
        vecfield_helmholtz!,vectorpotential_uniformvecfield!,scalarpotential_uniformvecfield!,vecfield_uniformvecfield!,
        create_CLinvCT,create_CL2invCT,create_GLinvD,create_nRTRn,create_RTLinvR,
        create_GLinvD_cross,create_CLinvCT_scalar,create_surface_filter,
        AreaRegionCache,LineRegionCache,PointRegionCache,arccoord,
        ForcingModelAndRegion,apply_forcing!,
        solve, @snapshotoutput,@scalarsurfacemetric,@vectorsurfacemetric,surfaces,
        prescribed_surface_jump!,prescribed_surface_average!,
        timestep


abstract type AbstractBasicCache{N,GCT} end


abstract type AbstractScalingType end

abstract type GridScaling <: AbstractScalingType end
abstract type IndexScaling <: AbstractScalingType end

abstract type LayerType{N} end

const DEFAULT_SCALING = GridScaling
const DEFAULT_DATA_TYPE = Float64
const DEFAULT_DDF = CartesianGrids.Yang3


"""
$(TYPEDEF)

A system of operators and caches for immersed layer problems. This is constructed
by [`construct_system`](@ref)
"""
mutable struct ILMSystem{static,PT,N,PHT,BCF,FF,DTF,MTF,BCT,ECT}

  phys_params :: PHT
  bc :: BCF
  forcing :: FF
  timestep_func :: DTF
  motions :: MTF
  base_cache :: BCT
  extra_cache :: ECT

end

_get_function_name(f) = f
_get_function_name(f::Symbol) = eval(f)


include("cartesian_extensions.jl")
include("tools.jl")
include("cache.jl")
include("problem.jl")
include("system.jl")
include("layers.jl")
include("grid_operators.jl")
include("surface_operators.jl")
include("forcing.jl")
include("matrix_operators.jl")
include("helmholtz.jl")
include("timemarching.jl")
include("output.jl")
include("bc.jl")

include("plot_recipes.jl")

@deprecate DoubleLayer surface_divergence!
@deprecate SingleLayer regularize!


end
