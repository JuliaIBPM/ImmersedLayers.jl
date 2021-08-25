# ImmersedLayers.jl

*tools for implementing discrete single and double layer potentials immersed in Cartesian grids*

The objective of this package is to implement
- the tools for immersing discretely-represented surfaces into Cartesian grids
- discrete Heaviside functions that mask the regions interior or exterior of surfaces
- discrete differential operators that immerse surface data into the grid (i.e.,
  "layers")
These operators and tools are described in detail in [^1].


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
