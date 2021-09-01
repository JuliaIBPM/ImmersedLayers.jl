# # Grid operations

#md # ```@meta
#md # CurrentModule = ImmersedLayers
#md # ```

#=
There are a variety of (purely) grid-based operators that are useful for carrying
out calculations in immersed layer problems.
We will start by generating the cache, just as we did in [Immersed layer caches](@ref)
=#


using ImmersedLayers
using CartesianGrids
using RigidBodyTools
using Plots
using LinearAlgebra

#md # ## Surface-grid operator functions
#md # ```@docs
#md # divergence!
#md # grad!
#md # curl!
#md # convective_derivative!
#md # convective_derivative
#md # ConvectiveDerivativeCache
#md # inverse_laplacian!
#md # ```
