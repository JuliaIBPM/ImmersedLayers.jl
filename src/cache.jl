"""
$(TYPEDEF)

Create a cache of operators and storage data for use in surface operations.

## Constructors
The cache is populated differently depending on the type of data intended. For
scalar surface quantities, one calls `SurfaceScalarCache`; for vector surface
quantities, one calls `SurfaceVectorCache`. The examples below can be replaced
with `SurfaceVectorCache` without modification.

`SurfaceScalarCache(X::VectorData,nrm::VectorData,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,weights=nothing])`

Here, `X` holds the coordinates of surface data and `nrm` holds the corresponding
surface normals.

The keyword `weights` can be used to set the weights in the regularization operation.
By default, it sets the regularization and interpolation to be symmetric matrices
(i.e., interpolation is the adjoint of regularization with respect to an inner product
  equal to a vector dot product). By using `weights = dsvec`, where `dsvec` is
  a vector of surface panel areas corresponding to `X` and `nrm`, then the
  interpolation and regularization are adjoints with respect to inner products
  based on discretized integrals.

`SurfaceScalarCache(body::Body/BodyList,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,weights=areas(body)])`

Here, the `body` can be of type `Body` or `BodyList`. The same keyword arguments
apply. However, now the default is `weights = areas(body)`.
"""
struct SurfaceCache{N,NT<:VectorData,REGT<:Regularize,RT<:RegularizationMatrix,ET<:InterpolationMatrix,
                      LT<:CartesianGrids.Laplacian,GVT,GNT,GCT,SVT,SST}
    nrm :: NT
    regop :: REGT
    R :: RT
    E :: ET
    L :: LT
    gv_cache :: GVT
    gv2_cache :: GVT
    gn_cache :: GNT
    gc_cache :: GCT
    sv_cache :: SVT
    sv2_cache :: SVT
    ss_cache :: SST
end

for f in [:SurfaceScalarCache, :SurfaceVectorCache]
  @eval $f(body::Union{Body,BodyList},g::PhysicalGrid;ddftype = CartesianGrids.Yang3, weights=areas(body).data) =
        $f(VectorData(collect(body)),normals(body),g;ddftype=ddftype,weights=weights)
end

function SurfaceScalarCache(X::VectorData{N},nrm::VectorData{N},g::PhysicalGrid; ddftype = CartesianGrids.Yang3, weights = nothing) where {N}

   sstmp = ScalarData(X)
   svtmp = VectorData(X)

   gvtmp = Edges(Primal,size(g))
   gntmp = Nodes(Dual,size(g))
   gctmp = Nodes(Primal,size(g))

   if isnothing(weights)
       regop = Regularize(X, cellsize(g), I0=origin(g), ddftype = CartesianGrids.Yang3, issymmetric=true)
       Rf, _ = RegularizationMatrix(regop, svtmp, gvtmp)
   else
       regop = Regularize(X, cellsize(g), I0=origin(g), ddftype = CartesianGrids.Yang3, weights=weights)
       Rf = RegularizationMatrix(regop, svtmp, gvtmp)
   end

   Ef = InterpolationMatrix(regop, gvtmp, svtmp)

   L = plan_laplacian(size(gntmp),with_inverse=true)
   return SurfaceCache{N,typeof(nrm),typeof(regop),typeof(Rf),typeof(Ef),typeof(L),
                        typeof(gvtmp),typeof(gntmp),typeof(gctmp),typeof(svtmp),typeof(sstmp)}(
                        nrm,regop,Rf,Ef,L,similar(gvtmp),similar(gvtmp),similar(gntmp),similar(gctmp),similar(svtmp),similar(svtmp),similar(sstmp))
end


function SurfaceVectorCache(X::VectorData{N},nrm::VectorData{N},g::PhysicalGrid; ddftype = CartesianGrids.Yang3, weights = nothing) where {N}

   sstmp = VectorData(X)
   svtmp = TensorData(X)

   gvtmp = EdgeGradient(Primal,size(g))
   gntmp = Nodes(Dual,size(g))
   gctmp = Edges(Primal,size(g))

   if isnothing(weights)
       regop = Regularize(X, cellsize(g), I0=origin(g), ddftype = CartesianGrids.Yang3, issymmetric=true)
       Rf, _ = RegularizationMatrix(regop, svtmp, gvtmp)
   else
       regop = Regularize(X, cellsize(g), I0=origin(g), ddftype = CartesianGrids.Yang3, weights=weights)
       Rf = RegularizationMatrix(regop, svtmp, gvtmp)
   end

   Ef = InterpolationMatrix(regop, gvtmp, svtmp)

   L = plan_laplacian(size(gntmp),with_inverse=true)
   return SurfaceCache{N,typeof(nrm),typeof(regop),typeof(Rf),typeof(Ef),typeof(L),
                        typeof(gvtmp),typeof(gntmp),typeof(gctmp),typeof(svtmp),typeof(sstmp)}(
                        nrm,regop,Rf,Ef,L,similar(gvtmp),similar(gvtmp),similar(gntmp),similar(gctmp),similar(svtmp),similar(svtmp),similar(sstmp))
end
