# # Matrix operators

#md # ```@meta
#md # CurrentModule = ImmersedLayers
#md # ```

#=
Many solutions of partial differential equations with immersed layers, particularly Poisson problems,
lead to saddle-point problems, in which the Schur complement operator is a matrix
composed from some of the surface-grid operators discussed in [Surface-grid operations](@ref).
The package provides some convenience tools for constructing these matrices. By
their nature, the construction of these matrices is slow, since each column
involves the application of the same set of operations. However, the point
of this construction is to do it once and store it for repeated application.
=#

#=
One common saddle point system is

$$A = \begin{bmatrix} L & R \\ R^T & 0 \end{bmatrix}$$

where $L$ is the discrete Laplacian and $R$ and $R^T$ are the regularization
and interpolation operators ([`regularize!`](@ref) and [`interpolate!`](@ref)),
respectively. This system arises in the solution of the Poisson equation
with Dirichlet boundary conditions on the immersed surface. The Schur complement of this is
$S = - R^T L^{-1} R$. This matrix can be obtained using the function [`create_RTLinvR`](@ref).
=#


#=
Another common saddle point system is

$$A = \begin{bmatrix} L & D_s \\ G_s & R_n^T R_n \end{bmatrix}$$

where $D_s$ and $G_s$ are the surface divergence and gradient operators
([`surface_divergence!`](@ref) and [`surface_grad!`](@ref)),
respectively, and $R_n$ and $R_n^T$ are [`regularize_normal!`](@ref) and [`normal_interpolate!`](@ref).
This system arises in the solution of the Poisson equation
with Neumann boundary conditions on the immersed surface. The Schur complement of this is
$S = R_n^T R_n - G_s L^{-1} D_s$. Each of the matrices in this are individually
provided by the package, by the functions [`create_nRTRn`](@ref) and [`create_GLinvD`](@ref),
respectively. However, it is useful to know that the sum of these two
matrices is exactly the matrix $-C_s L^{-1}C_s^T$, where $C_s$ and $C_s^T$ are
surface curl operators [`surface_curl!`](@ref). This complete matrix is provided by
[`create_CLinvCT`](@ref).
=#

#=
Another helpful matrix operator is the surface filter, given by

$$\tilde{R}^T R$$

where $\tilde{R}^T$ is a modified form of the interpolation operator,
designed to return the regularized field to the surface points while
maintaining the integral value of the original field [^1]. We can
obtain this matrix with [`create_surface_filter`](@ref).
=#

#md # ## Matrix construction functions

#md # ```@docs
#md # create_RTLinvR
#md # create_CLinvCT
#md # create_CL2invCT
#md # create_GLinvD
#md # create_GLinvD_cross
#md # create_nRTRn
#md # create_surface_filter
#md # ```

#md # [^1]: Goza, A., et al., (2016) "Accurate computation of surface stresses and forces with immersed boundary methods," J. Comput. Phys., 321, 860--873.
