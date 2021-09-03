# ImmersedLayers.jl
_Tools for immersing surfaces and their operations in Cartesian grids_

| Documentation | Build Status |
|:---:|:---:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaIBPM.github.io/ImmersedLayers.jl/stable) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaIBPM.github.io/ImmersedLayers.jl/dev) | [![Build Status](https://github.com/JuliaIBPM/ImmersedLayers.jl/workflows/CI/badge.svg)](https://github.com/JuliaIBPM/ImmersedLayers.jl/actions) [![Coverage](https://codecov.io/gh/JuliaIBPM/ImmersedLayers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaIBPM/ImmersedLayers.jl) |


## Package objective

The objective of this package is to implement
* the tools for regularizing and interpolating data between discretely-represented surfaces and Cartesian grids
* discrete Heaviside functions that mask the regions interior or exterior of surfaces
* discrete differential operators that immerse surface data into the grid (i.e.,
  "layers")

At this time, the package only implements these in two spatial dimensions. The operators and tools are described in detail in [^1], but a summary is given below.

## Background

A lot of problems in physics involve surfaces and their interaction with fields
in a higher-dimensional space. One way to facilitate this interaction
is by "immersing" the surface data, as well as the associated operations on these data, into the higher-dimensional space, and similarly, allowing data in the
higher-dimensional space to be restricted to the surface. Underlying these is
the concept of a "masked" field, which takes a certain continuous form on one
side of a surface and another form on the other side. A Heaviside function H can be used to write this mathematically [^1]:

f = H(χ) f⁺ + H(-χ) f⁻

where χ is a level set function, taking a positive value on the + side of the surface and negative value on the - side. The χ=0 level set implicitly defines the surface. Also, the gradient of χ is proportional to the unit normal vector, **n**. In fact, we can always choose this function so that it *is* the local normal.

A really neat thing happens when we take a spatial derivative of f. For example, the gradient:

∇f = H(χ) ∇f⁺  + H(-χ) ∇f⁻ + δ(χ)n(f⁺ - f⁻)

Then, we get a masked form of the gradient fields of f on either side, *plus* a term involving the jump in f across the surface, times the normal vector, times the Dirac delta function δ(χ). This last factor is the immersion operator: it immerses the surface jump in f into the higher-dimensional space. Other derivatives (e.g., curl, divergence of vector fields) lead to immersion similar terms. If we the divergence of the gradient above, we get a Poisson equation with two immersion terms:

∇²f = H(χ) ∇²f⁺ + H(-χ) ∇²f⁻ + δ(χ)n⋅(∇f⁺ - ∇f⁻) +  ∇⋅[δ(χ)n(f⁺ - f⁻)]

The last two terms are single and double layers, in the language of the theory of potentials. Generically, in any partial differential equation, we refer to these terms as **immersed layers**.

So standard partial differential equations can be adapted for the masked fields, so that the equations are augmented with the surface quantities. Restriction, δᵀ(χ), is the transpose of immersion. It arises when we wish to impose constraints on the surface behavior; we can instead apply this constraint to the restricted form of the masked field, e.g., setting it to a prescribed value, fᵦ, on the surface:

δᵀ(χ) f = fᵦ

In a computational environment, we discretize the fields on both the surface as well as in the higher-dimensional space, so this immersion process involves *regularization* (the discrete form of immersion) and *interpolation* (the discrete form of restriction), defined with the help of a discrete version of the Dirac delta function, the "DDF". We discretize the higher-dimensional space with the a staggered Cartesian grid, using tools in the `CartesianGrids.jl` package.

For example, to regularize surface scalar data to the cell centers of the grid,
we use a matrix operator, Rc. Alternatively, for regularizing vector surface data to cell faces, we use Rf. Each of these has a transpose, used for interpolation of the grid data to the surface points, e.g, Rcᵀ.

When we combine these operations with the standard differential operators on the grid, we get a powerful set of tools for numerically solving PDEs.


## Installation

This package works on Julia `1.4` and above and is registered in the general Julia registry. To install from the REPL, type
e.g.,
```julia
] add ImmersedLayers
```

Then, in any version, type
```julia
julia> using ImmersedLayers
```

The plots in this documentation are generated using [Plots.jl](http://docs.juliaplots.org/latest/).
You might want to install that, too, to follow the examples.

## References

[^1]: Eldredge, J. D. (2021) "A method of immersed layers on Cartesian grids, with application to incompressible flows," arXiv:2103.04521.


[1] Eldredge, J. D. (2021) "A method of immersed layers on Cartesian grids, with application to incompressible flows," [arXiv:2103.04521](https://arxiv.org/abs/2103.04521).
