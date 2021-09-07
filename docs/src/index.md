# ImmersedLayers.jl

*tools for immersing surfaces and their operations in Cartesian grids*

## Package objective

The objective of this package is to implement
- the tools for regularizing and interpolating data between discretely-represented surfaces and Cartesian grids
- discrete Heaviside functions that mask the regions interior or exterior of surfaces
- discrete differential operators that immerse surface data into the grid (i.e.,
  "layers")

At this time, the package only implements these in two spatial dimensions. The operators and tools are described in detail in [^1], but a summary is given below.

## Background

A lot of problems in physics involve surfaces and their interaction with fields
in a higher-dimensional space. One way to facilitate this interaction
is by "immersing" the surface data, as well as the associated operations on these data, into the higher-dimensional space, and similarly, allowing data in the
higher-dimensional space to be restricted to the surface. Underlying these is
the concept of a "masked" field, which takes a certain continuous form on one
side of a surface and another form on the other side. A Heaviside function $H$ can be used to write this mathematically [^1]:

$$f = H(\chi) f^+ + H(-\chi) f^-$$

where $\chi$ is a level set function, taking a positive value on the $+$ side of the surface and negative value on the $-$ side. The $\chi=0$ level set implicitly defines the surface. Also, the gradient of $\chi$ is proportional to the unit normal vector, $\mathbf{n}$. In fact, we can always choose this function so that it *is* the local normal.

A really neat thing happens when we take a spatial derivative of $f$. For example, the gradient:

$$\nabla f = H(\chi) \nabla f^+ + H(-\chi) \nabla f^- + \delta(\chi)\mathbf{n}(f^+ - f^-)$$

Then, we get a masked form of the gradient fields of $f$ on either side, *plus* a term involving the jump in $f$ across the surface, times the normal vector, times the Dirac delta function $\delta(\chi)$. This last factor is the immersion operator: it immerses the surface jump in $f$ into the higher-dimensional space. Other derivatives (e.g., curl, divergence of vector fields) lead to immersion similar terms. If we the divergence of the gradient above, we get a Poisson equation with two immersion terms:

$$\nabla^2 f = H(\chi) \nabla^2 f^+ + H(-\chi) \nabla^2 f^- + \delta(\chi)\mathbf{n} \cdot (\nabla f^+ - \nabla f^-) +  \nabla \cdot \left[\delta(\chi)\mathbf{n}(f^+ - f^-) \right]$$

The last two terms are single and double layers, in the language of the theory of potentials. Generically, in any partial differential equation, we refer to these terms as **immersed layers**.

So standard partial differential equations can be adapted for the masked fields, so that the equations are augmented with the surface quantities. Restriction, $\delta^T(\chi)$, is the transpose of immersion. It arises when we wish to impose constraints on the surface behavior; we can instead apply this constraint to the restricted form of the masked field, e.g., setting it to a prescribed value, $f_s$, on the surface:

$$\delta^T(\chi) f = f_s$$

In a computational environment, we discretize the fields on both the surface as well as in the higher-dimensional space, so this immersion process involves *regularization* (the discrete form of immersion) and *interpolation* (the discrete form of restriction), defined with the help of a discrete version of the Dirac delta function, the "DDF". We discretize the higher-dimensional space with the a staggered Cartesian grid, using tools in the `CartesianGrids.jl` package, which is
fully exported by this package. The package also exports `RigidBodyTools.jl`,
which has a variety of tools for creating and transforming bodies.

For example, to regularize surface scalar data to the cell centers of the grid,
we use a matrix operator, $R_c$. Alternatively, for regularizing vector surface data to cell faces, we use $R_f$. Each of these has a transpose, used for interpolation of the grid data to the surface points, e.g, $R_c^T$.

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
