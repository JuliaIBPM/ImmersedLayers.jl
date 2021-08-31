abstract type AbstractScalingType end

abstract type GridScaling <: AbstractScalingType end
abstract type IndexScaling <: AbstractScalingType end

"""
$(TYPEDEF)

When defining problem-specific cache, make it a subtype of this.
"""
abstract type AbstractExtraILMCache end


"""
$(TYPEDEF)

A cache of operators and storage data for use in surface operations. Constructed
with [`SurfaceScalarCache`](@ref) or [`SurfaceVectorCache`](@ref).
"""
struct BasicILMCache{N,SCA<:AbstractScalingType,ND,PT<:VectorData,NT<:VectorData,
                      DST<:ScalarData,REGT<:Regularize,
                      RSNT<:RegularizationMatrix,ESNT<:InterpolationMatrix,
                      RT<:RegularizationMatrix,ET<:InterpolationMatrix,
                      LT<:CartesianGrids.Laplacian,GVT,GNT,GCT,SVT,SST}

    g :: PhysicalGrid{ND}
    X :: PT
    nrm :: NT
    ds :: DST
    regop :: REGT
    Rsn :: RSNT
    Esn :: ESNT
    R :: RT
    E :: ET
    L :: LT
    gsnorm_cache :: GVT
    gsnorm2_cache :: GVT
    gcurl_cache :: GNT
    gdata_cache :: GCT
    snorm_cache :: SVT
    snorm2_cache :: SVT
    sdata_cache :: SST
end

for f in [:SurfaceScalarCache, :SurfaceVectorCache]
  @eval $f(body::Union{Body,BodyList},g::PhysicalGrid;ddftype = CartesianGrids.Yang3, scaling = IndexScaling) =
        $f(points(body),areas(body),normals(body),g;ddftype=ddftype,scaling=scaling)
end

"""
$(TYPEDEF)

Create a cache of type `BasicILMCache`, holding operators and storage data
for use in immersed layer operations on scalar data.

This is often only called from within`ILMSystem` rather than directly. If called
directly, there are two basic forms:

`SurfaceScalarCache(X::VectorData,A::ScalarData,nrm::VectorData,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=IndexScaling])`

Here, `X` holds the coordinates of surface data, `A` the areas, and `nrm` holds the corresponding
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
function SurfaceScalarCache(X::VectorData{N},a::ScalarData{N},nrm::VectorData{N},g::PhysicalGrid;
                              ddftype = CartesianGrids.Yang3,
                              scaling = IndexScaling) where {N}

   sdata_cache = ScalarData(X)
   snorm_cache = VectorData(X)

   gsnorm_cache = Edges(Primal,size(g))
   gcurl_cache = Nodes(Dual,size(g))
   gdata_cache = Nodes(Primal,size(g))

   _surfacecache(X,a,nrm,g,ddftype,scaling,sdata_cache,snorm_cache,gsnorm_cache,gcurl_cache,gdata_cache)

end

"""
$(TYPEDEF)

Create a cache of type `BasicILMCache`, holding operators and storage data
for use in immersed layer operations on vector data.

This is often only called from within`ILMSystem` rather than directly. If called
directly, there are two basic forms:

`SurfaceVectorCache(X::VectorData,A::ScalarData,nrm::VectorData,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=IndexScaling])`

Here, `X` holds the coordinates of surface data, `A` the areas, and `nrm` holds the corresponding
surface normals.

The keyword `scaling` can be used to set the scaling in the operations.
By default, it is set to `IndexScaling` which sets the regularization and interpolation to
be symmetric matrices (i.e., interpolation is the adjoint of regularization with
  respect to a vector dot product), and the vector calculus operations on the grid
  are simple differences. By using `scaling = GridScaling`, then the grid and
  point spacings are accounted for. Interpolation and regularization are adjoints
  with respect to inner products based on discretized surface and volume integrals,
  and vector calculus operations are scaled by the grid spacing.

`SurfaceVectorCache(body::Body/BodyList,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=IndexScaling])`

Here, the `body` can be of type `Body` or `BodyList`. The same keyword arguments
apply.
"""
function SurfaceVectorCache(X::VectorData{N},a::ScalarData{N},nrm::VectorData{N},g::PhysicalGrid;
                              ddftype = CartesianGrids.Yang3,
                              scaling = IndexScaling) where {N}

   sdata_cache = VectorData(X)
   snorm_cache = TensorData(X)

   gsnorm_cache = EdgeGradient(Primal,size(g))
   gcurl_cache = Nodes(Dual,size(g))
   gdata_cache = Edges(Primal,size(g))

   _surfacecache(X,a,nrm,g,ddftype,scaling,sdata_cache,snorm_cache,gsnorm_cache,gcurl_cache,gdata_cache)

end

function Base.show(io::IO, H::BasicILMCache{N,SCA}) where {N,SCA}
    println(io, "Surface cache with scaling of type $SCA")
    println(io, "  $N point data of type $(typeof(H.sdata_cache))")
    println(io, "  Grid data of type $(typeof(H.gdata_cache))")
end

function _surfacecache(X::VectorData{N},a,nrm,g::PhysicalGrid{ND},ddftype,scaling,sdata_cache,snorm_cache,gsnorm_cache,gcurl_cache,gdata_cache) where {N,ND}

  regop = _get_regularization(X,a,g,ddftype,scaling)
  Rsn = _regularization_matrix(regop,snorm_cache,gsnorm_cache)
  Esn = InterpolationMatrix(regop, gsnorm_cache, snorm_cache)

  R = _regularization_matrix(regop,sdata_cache,gdata_cache )
  E = InterpolationMatrix(regop, gdata_cache,sdata_cache)

  L = plan_laplacian(size(gcurl_cache),with_inverse=true)
  return BasicILMCache{N,scaling,ND,typeof(X),typeof(nrm),typeof(a),typeof(regop),typeof(Rsn),typeof(Esn),typeof(R),typeof(E),typeof(L),
                       typeof(gsnorm_cache),typeof(gcurl_cache),typeof(gdata_cache),typeof(snorm_cache),typeof(sdata_cache)}(
                       g,X,nrm,a,regop,Rsn,Esn,R,E,L,
                       similar(gsnorm_cache),similar(gsnorm_cache),similar(gcurl_cache),similar(gdata_cache),
                       similar(snorm_cache),similar(snorm_cache),similar(sdata_cache))

end

@inline CartesianGrids.cellsize(s::BasicILMCache) = cellsize(s.g)



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

## Obtaining copies of the grid and surface data
"""
    similar_grid(::BasicILMCache)

Get a `similar` copy of the basic grid data in the cache.
"""
@inline similar_grid(cache::BasicILMCache,kwargs...) = similar(cache.gdata_cache,kwargs...)

"""
    similar_gridcurl(::BasicILMCache)

Get a `similar` copy of the grid curl field data in the cache.
"""
@inline similar_gridcurl(cache::BasicILMCache,kwargs...) = similar(cache.gcurl_cache,kwargs...)


"""
    similar_surface(::BasicILMCache)

Get a `similar` copy of the basic surface point data in the cache.
"""
@inline similar_surface(cache::BasicILMCache,kwargs...) = similar(cache.sdata_cache,kwargs...)

"""
    zeros_grid(::BasicILMCache)

Get a copy of the basic grid data in the cache, with values set to zero.
"""
@inline zeros_grid(cache::BasicILMCache,kwargs...) = zero(cache.gdata_cache,kwargs...)

"""
    zeros_gridcurl(::BasicILMCache)

Get a copy of the grid curl field data in the cache, with values set to zero.
"""
@inline zeros_gridcurl(cache::BasicILMCache,kwargs...) = zero(cache.gcurl_cache,kwargs...)


"""
    zeros_surface(::BasicILMCache)

Get a copy of the basic surface point data in the cache, with values set to zero.
"""
@inline zeros_surface(cache::BasicILMCache,kwargs...) = zero(cache.sdata_cache,kwargs...)

"""
    ones_grid(::BasicILMCache)

Get a copy of the basic grid data in the cache, with values set to one.
"""
@inline ones_grid(cache::BasicILMCache,kwargs...) = ones(cache.gdata_cache,kwargs...)

"""
    ones_surface(::BasicILMCache)

Get a copy of the basic surface point data in the cache, with values set to one.
"""
@inline ones_surface(cache::BasicILMCache,kwargs...) = ones(cache.sdata_cache,kwargs...)


# Extend operators on body points
"""
    points(cache::BasicILMCache)

Return the coordinates (as `VectorData`) of the surface points associated with `cache`
"""
points(cache::BasicILMCache) = cache.X

"""
    normals(cache::BasicILMCache)

Return the normals (as `VectorData`) of the surface points associated with `cache`
"""
normals(cache::BasicILMCache) = cache.nrm

"""
    areas(cache::BasicILMCache)

Return the areas (as `ScalarData`) of the surface panels associated with `cache`
"""
areas(cache::BasicILMCache) = cache.ds

# Extend norms and inner products
"""
    norm(u::GridData,cache::BasicILMCache)

Calculate the norm of grid data `u`, using the scaling associated with `cache`.
""" norm(u::GridData,cache::BasicILMCache)

norm(u::GridData,cache::BasicILMCache{N,GridScaling}) where {N} = norm(u,cache.g)

norm(u::GridData,cache::BasicILMCache{N,IndexScaling}) where {N} = norm(u)

"""
    dot(u1::GridData,u2::GridData,cache::BasicILMCache)

Calculate the inner product of grid data `u1` and `u2`, using the scaling associated with `cache`.
""" dot(u1::GridData,u2::GridData,cache::BasicILMCache)

dot(u1::GridData,u2::GridData,cache::BasicILMCache{N,GridScaling}) where {N} = dot(u1,u2,cache.g)

dot(u1::GridData,u2::GridData,cache::BasicILMCache{N,IndexScaling}) where {N} = dot(u1,u2)

"""
    norm(u::PointData,cache::BasicILMCache)

Calculate the norm of surface point data `u`, using the scaling associated with `cache`.
""" norm(u::PointData,cache::BasicILMCache)

norm(u::PointData{N},cache::BasicILMCache{N,GridScaling}) where {N} = norm(u,cache.ds)

norm(u::PointData{N},cache::BasicILMCache{N,IndexScaling}) where {N} = norm(u)



for f in [:ScalarData,:VectorData]
  @eval dot(u1::$f{N},u2::$f{N},cache::BasicILMCache{N,GridScaling}) where {N} = dot(u1,u2,cache.ds)

  @eval dot(u1::$f{N},u2::$f{N},cache::BasicILMCache{N,IndexScaling}) where {N} = dot(u1,u2)
end
"""
    dot(u1::PointData,u2::PointData,cache::BasicILMCache)

Calculate the inner product of surface point data `u1` and `u2`, using the scaling associated with `cache`.
""" dot(u1::PointData,u2::PointData,cache::BasicILMCache)
