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
struct BasicILMCache{N,SCA<:AbstractScalingType,ND,BLT<:BodyList,NT<:VectorData,
                      DST<:ScalarData,REGT<:Regularize,
                      RSNT<:RegularizationMatrix,ESNT<:InterpolationMatrix,
                      RT<:RegularizationMatrix,ET<:InterpolationMatrix,
                      LT<:CartesianGrids.Laplacian,GVT,GNT,GCT,SVT,SST}

    g :: PhysicalGrid{ND}
    bl :: BLT
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
  @eval $f(body::Body,g::PhysicalGrid;ddftype = CartesianGrids.Yang3, scaling = IndexScaling) =
        $f(BodyList([body]),areas(body),normals(body),g;ddftype=ddftype,scaling=scaling)

  @eval $f(bl::BodyList,g::PhysicalGrid;ddftype = CartesianGrids.Yang3, scaling = IndexScaling) =
        $f(bl,areas(bl),normals(bl),g;ddftype=ddftype,scaling=scaling)

  @eval $f(g::PhysicalGrid;scaling = IndexScaling) =
        $f(BodyList(),ScalarData(0),VectorData(0),g,scaling=scaling)

  @eval function $f(X::VectorData,g::PhysicalGrid;ddftype = CartesianGrids.Yang3, scaling = IndexScaling)
          x = Vector{Float64}(undef,length(X.u))
          y = Vector{Float64}(undef,length(X.v))
          x .= X.u
          y .= X.v
          $f(BasicBody(x,y),g,ddftype=ddftype,scaling=scaling)
  end

end
"""
    SurfaceScalarCache(g::PhysicalGrid[,scaling=IndexScaling])

Create a cache of type `BasicILMCache` with scalar grid data, using the grid specified
in `g`, with no immersed points. The keyword `scaling`
can be used to set the scaling in the operations.
By default, it is set to `IndexScaling` which sets the differential operators
to be only differencing operators. By using `scaling = GridScaling`, then the grid and
 spacings are accounted for and differential operators are scaled by this spacing.
""" SurfaceScalarCache(::PhysicalGrid)

"""
    SurfaceVectorCache(g::PhysicalGrid[,scaling=IndexScaling])

Create a cache of type `BasicILMCache` with vector grid data, with no immersed points.
See [`SurfaceScalarCache`](@ref) for details.
""" SurfaceVectorCache(::PhysicalGrid)

"""
    SurfaceVectorCache(body::Body/BodyList,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=IndexScaling])

Create a cache of type `BasicILMCache`, holding operators and storage data
for use in immersed layer operations on vector data. See [`SurfaceScalarCache`](@ref)
for details.
""" SurfaceVectorCache(::Union{Body,BodyList},::PhysicalGrid)

"""
    SurfaceScalarCache(body::Body/BodyList,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=IndexScaling])

Create a cache of type `BasicILMCache`, holding operators and storage data
for use in immersed layer operations on scalar data. This is sometimes called from within`ILMSystem` rather than directly.

The `body` can be of type `Body` or `BodyList`. The keyword `scaling` can be used to set the scaling in the operations.
By default, it is set to `IndexScaling` which sets the regularization and interpolation to
be symmetric matrices (i.e., interpolation is the adjoint of regularization with
  respect to a vector dot product), and the vector calculus operations on the grid
  are simple differences. By using `scaling = GridScaling`, then the grid and
  point spacings are accounted for. Interpolation and regularization are adjoints
  with respect to inner products based on discretized surface and volume integrals,
  and vector calculus operations are scaled by the grid spacing.
""" SurfaceScalarCache(::Union{Body,BodyList},::PhysicalGrid)

"""
    SurfaceScalarCache(X::VectorData,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=IndexScaling])

Create a cache of type `BasicILMCache`, holding operators and storage data
for use in immersed layer operations on scalar data. The `X` specifies the
immersed point coordinates, and `g` the physical grid.
""" SurfaceScalarCache(::VectorData,::ScalarData,::VectorData,::PhysicalGrid)

"""
    SurfaceVectorCache(X::VectorData,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=IndexScaling])

Create a cache of type `BasicILMCache`, holding operators and storage data
for use in immersed layer operations on vector data. See [`SurfaceScalarCache`](@ref)
for details.
""" SurfaceVectorCache(::VectorData,::ScalarData,::VectorData,::PhysicalGrid)



function SurfaceScalarCache(bl::BodyList,a::ScalarData{N},nrm::VectorData{N},g::PhysicalGrid;
                              ddftype = CartesianGrids.Yang3,
                              scaling = IndexScaling) where {N}

   X = points(bl)
   sdata_cache = ScalarData(X)
   snorm_cache = VectorData(X)

   gsnorm_cache = Edges(Primal,size(g))
   gcurl_cache = Nodes(Dual,size(g))
   gdata_cache = Nodes(Primal,size(g))

   _surfacecache(bl,X,a,nrm,g,ddftype,scaling,sdata_cache,snorm_cache,gsnorm_cache,gcurl_cache,gdata_cache)

end



function SurfaceVectorCache(bl::BodyList,a::ScalarData{N},nrm::VectorData{N},g::PhysicalGrid;
                              ddftype = CartesianGrids.Yang3,
                              scaling = IndexScaling) where {N}

   X = points(bl)
   sdata_cache = VectorData(X)
   snorm_cache = TensorData(X)

   gsnorm_cache = EdgeGradient(Primal,size(g))
   gcurl_cache = Nodes(Dual,size(g))
   gdata_cache = Edges(Primal,size(g))

   _surfacecache(bl,X,a,nrm,g,ddftype,scaling,sdata_cache,snorm_cache,gsnorm_cache,gcurl_cache,gdata_cache)

end

function Base.show(io::IO, H::BasicILMCache{N,SCA}) where {N,SCA}
    println(io, "Surface cache with scaling of type $SCA")
    println(io, "  $N point data of type $(typeof(H.sdata_cache))")
    println(io, "  Grid data of type $(typeof(H.gdata_cache))")
end

function _surfacecache(bl::BodyList,X::VectorData{N},a,nrm,g::PhysicalGrid{ND},ddftype,scaling,sdata_cache,snorm_cache,gsnorm_cache,gcurl_cache,gdata_cache) where {N,ND}

  regop = _get_regularization(X,a,g,ddftype,scaling)
  Rsn = _regularization_matrix(regop,snorm_cache,gsnorm_cache)
  Esn = _interpolation_matrix(regop, gsnorm_cache, snorm_cache)

  R = _regularization_matrix(regop,sdata_cache,gdata_cache )
  E = _interpolation_matrix(regop, gdata_cache,sdata_cache)

  L = plan_laplacian(size(gcurl_cache),with_inverse=true)
  return BasicILMCache{N,scaling,ND,typeof(bl),typeof(nrm),typeof(a),typeof(regop),typeof(Rsn),typeof(Esn),typeof(R),typeof(E),typeof(L),
                       typeof(gsnorm_cache),typeof(gcurl_cache),typeof(gdata_cache),typeof(snorm_cache),typeof(sdata_cache)}(
                       g,bl,nrm,a,regop,Rsn,Esn,R,E,L,
                       similar(gsnorm_cache),similar(gsnorm_cache),similar(gcurl_cache),similar(gdata_cache),
                       similar(snorm_cache),similar(snorm_cache),similar(sdata_cache))

end

@inline CartesianGrids.cellsize(s::BasicILMCache) = cellsize(s.g)



# Standardize the regularization
_get_regularization(X::VectorData{N},a::ScalarData{N},g::PhysicalGrid,ddftype,::Type{GridScaling};filter=false) where {N} =
     Regularize(X,cellsize(g),I0=origin(g),weights=a.data,ddftype=ddftype,filter=filter)

_get_regularization(X::VectorData{N},a::ScalarData{N},g::PhysicalGrid,ddftype,::Type{IndexScaling};filter=false) where {N} =
     Regularize(X,cellsize(g),I0=origin(g),issymmetric=true,ddftype=ddftype,filter=filter)

_get_regularization(body::Union{Body,BodyList},args...;kwargs...) = _get_regularization(VectorData(collect(body)),areas(body),args...;kwargs...)

# This is needed to stabilize the type-unstable `RegularizationMatrix` function in
# CartesianGrids
function _regularization_matrix(regop::Regularize,src::PointData,trg::GridData)
    if regop._issymmetric
      R, _ = RegularizationMatrix(regop, src, trg)
    else
      R = RegularizationMatrix(regop, src, trg)
    end
    return R
end

@inline _interpolation_matrix(regop::Regularize,src::GridData,trg::PointData) =
        InterpolationMatrix(regop,src,trg)

# An API to generate regularization and interpolation matrices not generated
# in the basic cache
"""
    RegularizationMatrix(cache::BasicILMCache,src::PointData,trg::GridData)

Create a regularization matrix for regularizing point data of type `src` to
grid data of type `trg`. (Both `src` and `trg` must be appropriately sized
for the grid and points in `cache`.)
"""
CartesianGrids.RegularizationMatrix(cache::BasicILMCache,src::PointData,trg::GridData) =
    _regularization_matrix(cache.regop,src,trg)


"""
    InterpolationMatrix(cache::BasicILMCache,src::GridData,trg::PointData)

Create a interpolation matrix for regularizing grid data of type `src` to
point data of type `trg`. (Both `src` and `trg` must be appropriately sized
for the grid and points in `cache`.)
"""
CartesianGrids.InterpolationMatrix(cache::BasicILMCache,src::GridData,trg::PointData) =
    _interpolation_matrix(cache.regop,src,trg)

# Some utilities to get the DDF type of the cache
_ddf_type(::DDF{DT}) where {DT} = DT
_ddf_type(R::Regularize) = _ddf_type(R.ddf)
_ddf_type(cache::BasicILMCache) = _ddf_type(cache.regop)

# Getting the list of first indices for each body, for partitioning of surface vectors
_firstindices(b::Body) = [1,length(body)+1]
_firstindices(bl::BodyList) = [map(i -> first(getrange(bl,i)),1:length(bl)); numpts(bl)+1]

## Obtaining copies of the grid and surface data
"""
    similar_grid(::BasicILMCache)

Get a `similar` copy of the basic grid data in the cache.
"""
@inline similar_grid(cache::BasicILMCache,kwargs...) = similar(cache.gdata_cache,kwargs...)

"""
    similar_gridgrad(::BasicILMCache)

Get a `similar` copy of the gradient of the grid data in the cache.
"""
@inline similar_gridgrad(cache::BasicILMCache,kwargs...) = similar(cache.gsnorm_cache,kwargs...)


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

Get an instance of the basic grid data in the cache, with values set to zero.
"""
@inline zeros_grid(cache::BasicILMCache,kwargs...) = zero(cache.gdata_cache,kwargs...)

"""
    zeros_gridgrad(::BasicILMCache,dim)

Get an instance of the gradient of the grid data in the cache, , in direction `dim`,
with values set to zero. If the data are of type `TensorGridData`, then
`dim` takes values from 1 to 2^2.
"""
@inline zeros_gridgrad(cache::BasicILMCache,kwargs...) = zero(cache.gsnorm_cache,kwargs...)


"""
    zeros_gridcurl(::BasicILMCache)

Get an instance of the grid curl field data in the cache, with values set to zero.
"""
@inline zeros_gridcurl(cache::BasicILMCache,kwargs...) = zero(cache.gcurl_cache,kwargs...)


"""
    zeros_surface(::BasicILMCache)

Get an instance of the basic surface point data in the cache, with values set to zero.
"""
@inline zeros_surface(cache::BasicILMCache,kwargs...) = zero(cache.sdata_cache,kwargs...)

"""
    ones_grid(::BasicILMCache)

Get an instance of the basic grid data in the cache, with values set to unity.
"""
@inline ones_grid(cache::BasicILMCache,kwargs...) = ones(cache.gdata_cache,kwargs...)

"""
    ones_gridgrad(::BasicILMCache,dim)

Get an instance of the gradient of the grid data in the cache, in direction `dim`,
with values set to unity. If the data are of type `TensorGridData`, then
`dim` takes values from 1 to 2^2.
"""
@inline ones_gridgrad(cache::BasicILMCache,kwargs...) = ones(cache.gsnorm_cache,kwargs...)


"""
    ones_gridcurl(::BasicILMCache)

Get an instance of the grid curl field data in the cache, with values set to unity.
"""
@inline ones_gridcurl(cache::BasicILMCache,kwargs...) = ones(cache.gcurl_cache,kwargs...)


"""
    ones_surface(::BasicILMCache)

Get an instance of the basic surface point data in the cache, with values set to unity.
"""
@inline ones_surface(cache::BasicILMCache,kwargs...) = ones(cache.sdata_cache,kwargs...)

"""
    x_grid(::BasicILMCache)

Return basic grid data filled with the grid `x` coordinate
"""
function x_grid(cache::BasicILMCache)
    xc, _ = coordinates(cache.gdata_cache,cache.g)
    p = zeros_grid(cache)
    p .= xc
end

"""
    x_gridcurl(::BasicILMCache)

Return basic grid curl field data filled with the grid `x` coordinate
"""
function x_gridcurl(cache::BasicILMCache)
    xc, _ = coordinates(cache.gcurl_cache,cache.g)
    p = zeros_gridcurl(cache)
    p .= xc
end


"""
    y_grid(::BasicILMCache)

Return basic grid data filled with the grid `y` coordinate
"""
function y_grid(cache::BasicILMCache)
    _,yc = coordinates(cache.gdata_cache,cache.g)
    p = zeros_grid(cache)
    p .= yc'
end

"""
    y_gridcurl(::BasicILMCache)

Return basic grid curl field data filled with the grid `y` coordinate
"""
function y_gridcurl(cache::BasicILMCache)
    _,yc = coordinates(cache.gcurl_cache,cache.g)
    p = zeros_gridcurl(cache)
    p .= yc'
end


# Extend operators on body points
"""
    points(cache::BasicILMCache)

Return the coordinates (as `VectorData`) of the surface points associated with `cache`
"""
points(cache::BasicILMCache) = points(cache.bl)

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

"""
    norm(u::PointData,cache::BasicILMCache,i::Int)

Calculate the norm of surface point data `u`, using the scaling associated with `cache`,
for body `i` in the body list of `cache`.
""" norm(u::PointData,cache::BasicILMCache,i::Int)

norm(u::PointData{N},cache::BasicILMCache{N,GridScaling},i::Int) where {N} = norm(u,cache.ds,cache.bl,i)

norm(u::PointData{N},cache::BasicILMCache{N,IndexScaling},i::Int) where {N} = norm(u,cache.bl,i)


for f in [:ScalarData,:VectorData]
  @eval dot(u1::$f{N},u2::$f{N},cache::BasicILMCache{N,GridScaling}) where {N} = dot(u1,u2,cache.ds)

  @eval dot(u1::$f{N},u2::$f{N},cache::BasicILMCache{N,IndexScaling}) where {N} = dot(u1,u2)
end
"""
    dot(u1::PointData,u2::PointData,cache::BasicILMCache)

Calculate the inner product of surface point data `u1` and `u2`, using the scaling associated with `cache`.
""" dot(u1::PointData,u2::PointData,cache::BasicILMCache)

"""
    dot(u1::PointData,u2::PointData,cache::BasicILMCache,i)

Calculate the inner product of surface point data `u1` and `u2` for body
`i` in the cache `cache`, scaling as appropriate for this cache.
""" dot(u1::PointData,u2::PointData,cache::BasicILMCache,i::Int)

dot(u1::PointData{N},u2::PointData{N},cache::BasicILMCache{N,GridScaling},i::Int) where {N} = dot(u1,u2,cache.ds,cache.bl,i)

dot(u1::PointData{N},u2::PointData{N},cache::BasicILMCache{N,IndexScaling},i::Int) where {N} = dot(u1,u2,cache.bl,i)


## Integration
"""
    integrate(u::PointData,cache::BasicILMCache)

Calculate the discrete surface integral of data `u` on the immersed points in `cache`.
This uses trapezoidal rule quadrature. If `u` is `VectorData`, then this returns a vector of the integrals in
each coordinate direction. This operation produces the same effect,
regardless if `cache` is set up for `GridScaling` or `IndexScaling`. In both
cases, the surface element areas are used.
"""
@inline integrate(u::PointData{N},cache::BasicILMCache{N}) where {N} = integrate(u,cache.ds)


"""
    integrate(u::PointData,cache::BasicILMCache,i::Int)

Calculate the discrete surface integral of scalar data `u` on the immersed points in `cache`,
on body `i` in the body list in `cache`.
This uses trapezoidal rule quadrature. If `u` is `VectorData`, then this returns a vector of the integrals in
each coordinate direction. This operation produces the same effect,
regardless if `cache` is set up for `GridScaling` or `IndexScaling`. In both
cases, the surface element areas are used.
"""
@inline integrate(u::PointData{N},cache::BasicILMCache{N},i::Int) where {N} = integrate(u,cache.ds,cache.bl,i)


# Extending some operations on body lists to the enclosing cache
"""
    view(u::PointData,cache::BasicILMCache,i::Int)

Provide a `view` of point data `u` corresponding to body `i` in the
list of bodies in `cache`.
"""
view(u::PointData,cache::BasicILMCache,i::Int) = view(u,cache.bl,i)

"""
    copyto!(u::PointData,v::PointData,cache::BasicILMCache,i::Int)

Copy the data in the elements of `v` associated with body `i` in the body list in `cache` to
the corresponding elements in `u`. These data must be of the same type (e.g.,
`ScalarData` or `VectorData`) and have the same length.
""" copyto!(u::PointData,v::PointData,cache::BasicILMCache,i::Int)

"""
    copyto!(u::ScalarData,v::AbstractVector,cache::BasicILMCache,i::Int)

Copy the data in `v` to the elements in `u` associated with body `i` in the body list in `cache`.
`v` must have the same length as this subarray of `u` associated with `i`.
""" copyto!(u::ScalarData,v::AbstractVector,cache::BasicILMCache,i::Int)

@inline copyto!(u::PointData{N},v,cache::BasicILMCache{N},i::Int) where {N} = copyto!(u,v,cache.bl,i)
