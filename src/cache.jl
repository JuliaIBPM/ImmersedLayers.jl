abstract type AbstractScalingType end

abstract type GridScaling <: AbstractScalingType end
abstract type IndexScaling <: AbstractScalingType end


"""
$(TYPEDEF)

Create a cache of operators and storage data for use in surface operations.

## Constructors
The cache is populated differently depending on the type of data intended. For
scalar surface quantities, one calls `SurfaceScalarCache`; for vector surface
quantities, one calls `SurfaceVectorCache`. The examples below can be replaced
with `SurfaceVectorCache` without modification.

`SurfaceScalarCache(X::VectorData,nrm::VectorData,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=IndexScaling])`

Here, `X` holds the coordinates of surface data and `nrm` holds the corresponding
surface normals.

The keyword `scaling` can be used to set the scaling in the operations.
By default, it is set to `IndexScaling` which sets the regularization and interpolation to
be symmetric matrices (i.e., interpolation is the adjoint of regularization with
  respect to a vector dot product), and the vector calculus operations on the grid
  are simple differences. By using `scaling = GridScaling`, then the grid and
  point spacings are accounted for. Interpolation and regularization are adjoints
  with respect to inner products based on discretized surface and volume integrals,
  and vector calculus operations are scaled by the grid spacing.

`SurfaceScalarCache(body::Body/BodyList,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=IndexScaling])`

Here, the `body` can be of type `Body` or `BodyList`. The same keyword arguments
apply.
"""
struct SurfaceCache{N,SCA<:AbstractScalingType,ND,NT<:VectorData,REGT<:Regularize,RT<:RegularizationMatrix,ET<:InterpolationMatrix,
                      LT<:CartesianGrids.Laplacian,GVT,GNT,GCT,SVT,SST}

    g :: PhysicalGrid{ND}
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
  @eval $f(body::Union{Body,BodyList},g::PhysicalGrid;ddftype = CartesianGrids.Yang3, scaling = IndexScaling) =
        $f(VectorData(collect(body)),areas(body),normals(body),g;ddftype=ddftype,scaling=scaling)
end

function SurfaceScalarCache(X::VectorData{N},a::ScalarData{N},nrm::VectorData{N},g::PhysicalGrid; ddftype = CartesianGrids.Yang3, scaling = IndexScaling) where {N}

   sstmp = ScalarData(X)
   svtmp = VectorData(X)

   gvtmp = Edges(Primal,size(g))
   gntmp = Nodes(Dual,size(g))
   gctmp = Nodes(Primal,size(g))

   _surfacecache(X,a,nrm,g,ddftype,scaling,sstmp,svtmp,gvtmp,gntmp,gctmp)

end


function SurfaceVectorCache(X::VectorData{N},a::ScalarData{N},nrm::VectorData{N},g::PhysicalGrid; ddftype = CartesianGrids.Yang3, scaling = IndexScaling) where {N}

   sstmp = VectorData(X)
   svtmp = TensorData(X)

   gvtmp = EdgeGradient(Primal,size(g))
   gntmp = Nodes(Dual,size(g))
   gctmp = Edges(Primal,size(g))

   _surfacecache(X,a,nrm,g,ddftype,scaling,sstmp,svtmp,gvtmp,gntmp,gctmp)

end

function _surfacecache(X::VectorData{N},a,nrm,g::PhysicalGrid{ND},ddftype,scaling,sstmp,svtmp,gvtmp,gntmp,gctmp) where {N,ND}

  regop = _get_regularization(X,a,g,ddftype,scaling)
  R = _regularization_matrix(regop,svtmp,gvtmp)
  E = InterpolationMatrix(regop, gvtmp, svtmp)

  L = plan_laplacian(size(gntmp),with_inverse=true)
  return SurfaceCache{N,scaling,ND,typeof(nrm),typeof(regop),typeof(R),typeof(E),typeof(L),
                       typeof(gvtmp),typeof(gntmp),typeof(gctmp),typeof(svtmp),typeof(sstmp)}(
                       g,nrm,regop,R,E,L,
                       similar(gvtmp),similar(gvtmp),similar(gntmp),similar(gctmp),
                       similar(svtmp),similar(svtmp),similar(sstmp))

end

@inline CartesianGrids.cellsize(s::SurfaceCache) = cellsize(s.g)



# Standardize the regularization
_get_regularization(X::VectorData{N},a::ScalarData{N},g::PhysicalGrid,ddftype,::Type{GridScaling}) where {N} =
     Regularize(X,cellsize(g),I0=origin(g),weights=a.data,ddftype=ddftype)

_get_regularization(X::VectorData{N},a::ScalarData{N},g::PhysicalGrid,ddftype,::Type{IndexScaling}) where {N} =
     Regularize(X,cellsize(g),I0=origin(g),issymmetric=true,ddftype=ddftype)

_get_regularization(body::Union{Body,BodyList},args...) = _get_regularization(VectorData(collect(body)),areas(body),args...)

# This is needed to stabilize the type-unstable `RegularizationMatrix` function in
# CartesianGrids
function _regularization_matrix(regop::Regularize,src,trg)
    if regop._issymmetric
      R, _ = RegularizationMatrix(regop, src, trg)
    else
      R = RegularizationMatrix(regop, src, trg)
    end
    return R
end
