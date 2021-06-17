"""
$(TYPEDEF)

Create a cache of operators and storage data for use in surface operations.

## Constructors
`SurfaceCache(X::VectorData,nrm::VectorData,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,weights=nothing])`

Here, `X` holds the coordinates of surface data and `nrm` holds the corresponding
surface normals.

The keyword `weights` can be used to set the weights in the regularization operation.
By default, it sets the regularization and interpolation to be symmetric matrices
(i.e., interpolation is the adjoint of regularization with respect to an inner product
  equal to a vector dot product). By using `weights = dsvec`, where `dsvec` is
  a vector of surface panel areas corresponding to `X` and `nrm`, then the
  interpolation and regularization are adjoints with respect to inner products
  based on discretized integrals.

`SurfaceCache(body::Body/BodyList,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,weights=areas(body)])`

Here, the `body` can be of type `Body` or `BodyList`. The same keyword arguments
apply. However, now the default is `weights = areas(body)`.
"""
struct SurfaceCache{N,NT<:VectorData,REGT<:Regularize,RT<:RegularizationMatrix,ET<:InterpolationMatrix,
                      LT<:CartesianGrids.Laplacian,GVT<:Edges{Primal},GNT<:Nodes{Dual},GCT<:Nodes{Primal},
                      SVT<:VectorData,SST<:ScalarData}
    nrm :: NT
    regop :: REGT
    R :: RT
    E :: ET
    L :: LT
    gv_cache :: GVT
    gn_cache :: GNT
    gc_cache :: GCT
    sv_cache :: SVT
    ss_cache :: SST
end

function SurfaceCache(X::VectorData{N},nrm::VectorData{N},g::PhysicalGrid; ddftype = CartesianGrids.Yang3, weights = nothing) where {N}

   σtmp = ScalarData(X)
   svtmp = VectorData(X)

   qtmp = Edges(Primal,size(g))
   wtmp = Nodes(Dual,size(g))
   ptmp = Nodes(Primal,size(g))

   if isnothing(weights)
       regop = Regularize(X, cellsize(g), I0=origin(g), ddftype = CartesianGrids.Yang3, issymmetric=true)
       Rf, _ = RegularizationMatrix(regop, svtmp, qtmp)
   else
       regop = Regularize(X, cellsize(g), I0=origin(g), ddftype = CartesianGrids.Yang3, weights=weights)
       Rf = RegularizationMatrix(regop, svtmp, qtmp)
   end

   Ef = InterpolationMatrix(regop, qtmp, svtmp)

   L = plan_laplacian(size(wtmp),with_inverse=true)
   return SurfaceCache{N,typeof(nrm),typeof(regop),typeof(Rf),typeof(Ef),typeof(L),
                        typeof(qtmp),typeof(wtmp),typeof(ptmp),typeof(svtmp),typeof(σtmp)}(
                        nrm,regop,Rf,Ef,L,similar(qtmp),similar(wtmp),similar(ptmp),similar(svtmp),similar(σtmp))
end

@inline SurfaceCache(body::Union{Body,BodyList},g::PhysicalGrid;ddftype = CartesianGrids.Yang3, weights=areas(body).data) = SurfaceCache(VectorData(collect(body)),normals(body),g;ddftype=ddftype,weights=weights)
