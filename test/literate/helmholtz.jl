# # Helmholtz decomposition

#md # ```@meta
#md # CurrentModule = ImmersedLayers
#md # ```

#=
For vector fields on the grid, we often prefer to work with potentials
of that field and compose the vector field from these potentials. This
is the so-called Helmholtz decomposition

$$\mathbf{v} = \nabla\phi + \nabla\times\psi$$

where $\psi$ is a vector potential (or streamfunction in two dimensions)
and $\phi$ is a scalar potential. Note that, in an unbounded domain, $\phi$
satisfies a Poisson equation:

$$\nabla^2\phi = \nabla\cdot\mathbf{v}$$

and, assuming that $\psi$ has zero 'gauge' (i.e., its divergence is zero), then
$\psi$ also satisfies a Poisson equation:

$$\nabla^2\psi = -\nabla\times\mathbf{v}$$

So we can recover the velocity field from the divergence and curl of the
vector field (in fluid dynamics, these are the rate of dilatation and the vorticity,
respectively) by solving these equations for the potentials and then
evaluating their gradient and curl.

With immersed layers, these are extendable to domains with surfaces
and their associated jump in the vector field $[\mathbf{v}] = \mathbf{v}^+ - \mathbf{v}^-$.
The crucial relationships are

$$\nabla\cdot{\overline{\mathbf{v}}} = \overline{\nabla\cdot\mathbf{v}} + \delta(\chi)\mathbf{n}\cdot[\mathbf{v}]$$

and

$$\nabla\times{\overline{\mathbf{v}}} = \overline{\nabla\times\mathbf{v}} + \delta(\chi)\mathbf{n}\times[\mathbf{v}]$$

These are, respectively, the divergence and curl of the masked vector field, $\overline{\mathbf{v}$.
In `ImmersedLayers.jl`, we are always working with the masked form of a field,
whether we acknowledge it or not. In contrast, the first terms on the right-hand sides of these are the masked divergence and curl
of the vector field in each side of the surface. The difference between these is
a type of single-layer surface term---a 'source sheet' or a 'vortex sheet', respectively---
due to the jump in vector field. If we omit the overbar notation, then the Helmholtz
decomposition and the Poisson equations all still hold.

This package has a number of routines that can be used to perform the
operations described here. They rely on two caches, [`ScalarPotentialCache`](@ref)
and [`VectorPotentialCache`](@ref), each of which supports one of the
respective potentials. The function [`vecfield_helmholtz!`](@ref) carries
out all of these operations in order to assemble the vector field $\mathbf{v}$
from the masked divergence and curl fields and the jump in the vector field.
(We can also add an additional irrotational, divergence-free vector field during
this assembly.
=#


#md # ## Helmholtz decomposition functions

#md # ```@docs
#md # ScalarPotentialCache
#md # VectorPotentialCache
#md # vectorpotential_from_masked_curlv!
#md # scalarpotential_from_masked_divv!
#md # vectorpotential_from_curlv!
#md # vecfield_from_vectorpotential!
#md # masked_curlv_from_curlv_masked!
#md # scalarpotential_from_divv!
#md # masked_divv_from_divv_masked!
#md # vecfield_from_scalarpotential!
#md # vecfield_helmholtz!
#md # vectorpotential_uniformvecfield!
#md # scalarpotential_uniformvecfield!
#md # vecfield_uniformvecfield!
#md # ```
