# # Problems and the system

#md # ```@meta
#md # CurrentModule = ImmersedLayers
#md # ```

#=
In specific problems that we wish to solve with immersed layers, there may
be other data and operators that we would like to cache. We do this with
an *extra cache*, which the user can define, along with a problem type associated
with this cache. The basic cache and the extra cache are generated and associated
together in a *system*.
=#

# ## Example

using ImmersedLayers
using CartesianGrids
using RigidBodyTools
using Plots


#md # ## Problem types and functions
#md #
#md # ```@docs
#md # AbstractScalarILMProblem
#md # AbstractVectorILMProblem
#md # BasicScalarILMProblem
#md # BasicVectorILMProblem
#md # prob_cache
#md # ```

#md # ## System types and functions
#md #
#md # ```@docs
#md # ILMSystem
#md # __init
#md # ```
