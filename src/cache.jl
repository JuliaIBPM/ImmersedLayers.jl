
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
struct BasicILMCache{N,SCA<:AbstractScalingType,GCT,ND,BLT<:BodyList,NT<:VectorData,
                      DST<:ScalarData,REGT<:Regularize,
                      RSNT<:RegularizationMatrix,ESNT<:InterpolationMatrix,
                      RT<:RegularizationMatrix,ET<:InterpolationMatrix,
                      RCT, ECT, RDT, EDT,
                      LT<:CartesianGrids.Laplacian,GVT,GNT,GDT,SVT,SDT,SST} <: AbstractBasicCache{N,GCT}

    # Grid
    g :: PhysicalGrid{ND}

    # Bodies
    bl :: BLT

    # Points
    pts :: NT

    # Normals
    nrm :: NT

    # Areas
    ds :: DST

    # Regularization operator
    regop :: REGT

    # Regularization and interpolation of tensor product data (gsnorm/snorm)
    Rsn :: RSNT
    Esn :: ESNT

    # Regularization and interpolation of basic data (gdata/sdata)
    R :: RT
    E :: ET

    # Regularization and interpolation of grid curl data (gcurl/sdata)
    Rcurl :: RCT
    Ecurl :: ECT

    # Regularization and interpolation of grid div data (gdiv/sdata)
    Rdiv :: RDT
    Ediv :: EDT

    # Laplacian (with no coefficient)
    L :: LT

    # For holding the grid data comprised of tensor product of basic grid data
    # and regularized normals (e.g., scalar -> Edges{Primal}, vector -> EdgeGradient)
    gsnorm_cache :: GVT
    gsnorm2_cache :: GVT

    # For holding the curl of the basic grid data (always Nodes{Dual})
    gcurl_cache :: GNT

    # For holding the div of the basic grid data (always Nodes{Primal})
    gdiv_cache :: GDT

    # For holding the basic data type (e.g., scalar -> Nodes{Primal}, vector -> Edges{Primal})
    gdata_cache :: GCT

    # For holding grid coordinates
    xg :: GCT
    yg :: GCT

    # For holding the surface data comprised of tensor product of basic data
    # and normals (e.g, scalar -> VectorData, vector -> TensorData)
    snorm_cache :: SVT
    snorm2_cache :: SVT

    # For holding the basic surface data (e.g., scalar -> ScalarData, vector -> VectorData)
    sdata_cache :: SDT

    # For holding surface scalar data (always ScalarData)
    sscalar_cache :: SST
end

RigidBodyTools.numpts(::BasicILMCache{N}) where {N} = N
RigidBodyTools.numpts(::AbstractBasicCache{N}) where {N} = N
scalingtype(::BasicILMCache{N,SCA}) where {N,SCA} = SCA
cache_datatype(::BasicILMCache{N,SCA,GCT}) where {N,SCA,GCT} = GCT <: Nodes ? :scalar : :vector
cache_datatype(::AbstractBasicCache{N,GCT}) where {N,GCT} = GCT <: Nodes ? :scalar : :vector
type_curlreg(::BasicILMCache{N,SCA,GCT,ND,BLT,NT,DST,REGT,RSNT,ESNT,RT,ET,RCT,ECT,RDT,EDT,LT,GVT,GNT,GDT,SVT,SDT,SST}) where {N,SCA,GCT,ND,BLT,NT,DST,REGT,RSNT,ESNT,RT,ET,RCT,ECT,RDT,EDT,LT,GVT,GNT,GDT,SVT,SDT,SST} = RCT
type_divreg(::BasicILMCache{N,SCA,GCT,ND,BLT,NT,DST,REGT,RSNT,ESNT,RT,ET,RCT,ECT,RDT,EDT,LT,GVT,GNT,GDT,SVT,SDT,SST}) where {N,SCA,GCT,ND,BLT,NT,DST,REGT,RSNT,ESNT,RT,ET,RCT,ECT,RDT,EDT,LT,GVT,GNT,GDT,SVT,SDT,SST} = RDT
type_sdata(::BasicILMCache{N,SCA,GCT,ND,BLT,NT,DST,REGT,RSNT,ESNT,RT,ET,RCT,ECT,RDT,EDT,LT,GVT,GNT,GDT,SVT,SDT,SST}) where {N,SCA,GCT,ND,BLT,NT,DST,REGT,RSNT,ESNT,RT,ET,RCT,ECT,RDT,EDT,LT,GVT,GNT,GDT,SVT,SDT,SST} = SDT
type_sscalar(::BasicILMCache{N,SCA,GCT,ND,BLT,NT,DST,REGT,RSNT,ESNT,RT,ET,RCT,ECT,RDT,EDT,LT,GVT,GNT,GDT,SVT,SDT,SST}) where {N,SCA,GCT,ND,BLT,NT,DST,REGT,RSNT,ESNT,RT,ET,RCT,ECT,RDT,EDT,LT,GVT,GNT,GDT,SVT,SDT,SST} = SST




"""
    SurfaceScalarCache(g::PhysicalGrid[,scaling=IndexScaling])

Create a cache of type `BasicILMCache` with scalar grid data, using the grid specified
in `g`, with no immersed points. The keyword `scaling`
can be used to set the scaling in the operations.
By default, it is set to `IndexScaling` which sets the differential operators
to be only differencing operators. By using `scaling = GridScaling`, then the grid and
 spacings are accounted for and differential operators are scaled by this spacing.
 The keyword `phys_params` can be used to supply physical parameters.
""" SurfaceScalarCache(::PhysicalGrid)

"""
    SurfaceVectorCache(g::PhysicalGrid[,scaling=IndexScaling])

Create a cache of type `BasicILMCache` with vector grid data, with no immersed points.
See [`SurfaceScalarCache`](@ref) for details.
""" SurfaceVectorCache(::PhysicalGrid)

"""
    SurfaceVectorCache(body::Body/BodyList,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=Gridcaling])

Create a cache of type `BasicILMCache`, holding operators and storage data
for use in immersed layer operations on vector data. See [`SurfaceScalarCache`](@ref)
for details.
""" SurfaceVectorCache(::Union{Body,BodyList},::PhysicalGrid)

"""
    SurfaceScalarCache(body::Body/BodyList,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=GridScaling])

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
    SurfaceScalarCache(X::VectorData,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=GridScaling])

Create a cache of type `BasicILMCache`, holding operators and storage data
for use in immersed layer operations on scalar data. The `X` specifies the
is assumed to hold the endpoints of the immersed surface segments, and `g` the physical grid.
""" SurfaceScalarCache(::VectorData,::ScalarData,::VectorData,::PhysicalGrid)

"""
    SurfaceVectorCache(X::VectorData,g::PhysicalGrid[,ddftype=CartesianGrids.Yang3][,scaling=GridScaling])

Create a cache of type `BasicILMCache`, holding operators and storage data
for use in immersed layer operations on vector data. See [`SurfaceScalarCache`](@ref)
for details.
""" SurfaceVectorCache(::VectorData,::ScalarData,::VectorData,::PhysicalGrid)



function SurfaceScalarCache(bl::BodyList,a::ScalarData{N},nrm::VectorData{N},g::PhysicalGrid;
                              ddftype = DEFAULT_DDF,
                              scaling = DEFAULT_SCALING,
                              dtype = DEFAULT_DATA_TYPE) where {N}

	X = points(bl)
  sscalar_cache = nothing
	sdata_cache = ScalarData(X, dtype = dtype)
	snorm_cache = VectorData(X, dtype = dtype)

	gsnorm_cache = Edges(Primal,size(g), dtype = dtype)
	gdata_cache = Nodes(Primal,size(g), dtype = dtype)
  gcurl_cache = Nodes(Dual,size(g), dtype = dtype)
  gdiv_cache = nothing #Nodes(Primal,size(g), dtype = dtype)

	_surfacecache(bl,X,a,nrm,g,ddftype,scaling,sdata_cache,snorm_cache,sscalar_cache,gsnorm_cache,gcurl_cache,gdiv_cache,gdata_cache;dtype=dtype)

end

function SurfaceScalarCache(bl::BodyList,a::ScalarData{N},nrm::VectorData{N},g::PhysicalGrid,L::Laplacian;
    ddftype = DEFAULT_DDF,
    scaling = DEFAULT_SCALING,
    dtype = DEFAULT_DATA_TYPE) where {N}

    X = points(bl)
    sscalar_cache = nothing
    sdata_cache = ScalarData(X, dtype = dtype)
    snorm_cache = VectorData(X, dtype = dtype)

    gsnorm_cache = Edges(Primal,size(g), dtype = dtype)
    gdata_cache = Nodes(Primal,size(g), dtype = dtype)
    gcurl_cache = Nodes(Dual,size(g), dtype = dtype)
    gdiv_cache = nothing #Nodes(Primal,size(g), dtype = dtype)

    _surfacecache(bl,X,a,nrm,g,ddftype,scaling,sdata_cache,snorm_cache,sscalar_cache,gsnorm_cache,gcurl_cache,gdiv_cache,gdata_cache,L;dtype=dtype)

end

function SurfaceVectorCache(bl::BodyList,a::ScalarData{N},nrm::VectorData{N},g::PhysicalGrid;
                              ddftype = DEFAULT_DDF,
                              scaling = DEFAULT_SCALING,
                              dtype = DEFAULT_DATA_TYPE) where {N}

	X = points(bl)
  sscalar_cache = ScalarData(X, dtype = dtype)
	sdata_cache = VectorData(X, dtype = dtype)
	snorm_cache = TensorData(X, dtype = dtype)

	gsnorm_cache = EdgeGradient(Primal,size(g), dtype = dtype)
  gdata_cache = Edges(Primal,size(g), dtype = dtype)
	gcurl_cache = Nodes(Dual,size(g), dtype = dtype)
  gdiv_cache = Nodes(Primal,size(g), dtype = dtype)

	_surfacecache(bl,X,a,nrm,g,ddftype,scaling,sdata_cache,snorm_cache,sscalar_cache,gsnorm_cache,gcurl_cache,gdiv_cache,gdata_cache;dtype=dtype)

end

function SurfaceVectorCache(bl::BodyList,a::ScalarData{N},nrm::VectorData{N},g::PhysicalGrid, L::Laplacian;
    ddftype = DEFAULT_DDF,
    scaling = DEFAULT_SCALING,
    dtype = DEFAULT_DATA_TYPE) where {N}

    X = points(bl)
    sscalar_cache = ScalarData(X, dtype = dtype)
    sdata_cache = VectorData(X, dtype = dtype)
    snorm_cache = TensorData(X, dtype = dtype)

    gsnorm_cache = EdgeGradient(Primal,size(g), dtype = dtype)
    gdata_cache = Edges(Primal,size(g), dtype = dtype)
    gcurl_cache = Nodes(Dual,size(g), dtype = dtype)
    gdiv_cache = Nodes(Primal,size(g), dtype = dtype)

    _surfacecache(bl,X,a,nrm,g,ddftype,scaling,sdata_cache,snorm_cache,sscalar_cache,gsnorm_cache,gcurl_cache,gdiv_cache,gdata_cache,L;dtype=dtype)

end

for f in [:SurfaceScalarCache, :SurfaceVectorCache]
  @eval $f(body::Body,g::PhysicalGrid; kwargs...) =
        $f(BodyList([body]),areas(body),normals(body),g; kwargs...)
    
  @eval $f(body::Body,g::PhysicalGrid,L::Laplacian; kwargs...) =
        $f(BodyList([body]),areas(body),normals(body),g,L; kwargs...)

  @eval $f(bl::BodyList,g::PhysicalGrid; kwargs...) =
        $f(bl,areas(bl),normals(bl),g; kwargs...)

  @eval $f(bl::BodyList,g::PhysicalGrid,L::Laplacian; kwargs...) =
        $f(bl,areas(bl),normals(bl),g,L; kwargs...)

  @eval $f(g::PhysicalGrid;kwargs...) =
        $f(BodyList(),ScalarData(0),VectorData(0),g; kwargs...)

  @eval $f(g::PhysicalGrid,L::Laplacian; kwargs...) =
        $f(BodyList(),ScalarData(0),VectorData(0),g,L; kwargs...)

  @eval function $f(X::VectorData,g::PhysicalGrid; kwargs...)
          x = Vector{Float64}(undef,length(X.u))
          y = Vector{Float64}(undef,length(X.v))
          x .= X.u
          y .= X.v
          $f(BasicBody(x,y),g; kwargs...)
  end

  @eval function $f(X::VectorData,g::PhysicalGrid; L::Laplacian,kwargs...)
    x = Vector{Float64}(undef,length(X.u))
    y = Vector{Float64}(undef,length(X.v))
    x .= X.u
    y .= X.v
    $f(BasicBody(x,y),g,L;kwargs...)
end

end

function Base.show(io::IO, H::BasicILMCache{N,SCA}) where {N,SCA}
    println(io, "Surface cache with scaling of type $SCA")
    println(io, "  $N point data of type $(typeof(H.sdata_cache))")
    println(io, "  Grid data of type $(typeof(H.gdata_cache))")
end

function _surfacecache(bl::BodyList,X::VectorData{N},a,nrm,g::PhysicalGrid{ND},ddftype,scaling,
                      sdata_cache,snorm_cache,sscalar_cache,gsnorm_cache,gcurl_cache,gdiv_cache,gdata_cache;dtype=Float64) where {N,ND}


  regop = _get_regularization(X,a,g,ddftype,scaling)
  Rsn = _regularization_matrix(regop,snorm_cache,gsnorm_cache)
  Esn = _interpolation_matrix(regop, gsnorm_cache, snorm_cache)

  R = _regularization_matrix(regop,sdata_cache,gdata_cache )
  E = _interpolation_matrix(regop, gdata_cache,sdata_cache)

  Rcurl = _regularization_matrix(regop,sscalar_cache,gcurl_cache )
  Ecurl = _interpolation_matrix(regop, gcurl_cache,sscalar_cache)

  Rdiv = _regularization_matrix(regop,sscalar_cache,gdiv_cache )
  Ediv = _interpolation_matrix(regop, gdiv_cache,sscalar_cache)

  coeff_factor = 1.0
  with_inverse = true
  L = _get_laplacian(coeff_factor,g,with_inverse,scaling;dtype=dtype)

  xg = _x_grid(gdata_cache,g)
  yg = _y_grid(gdata_cache,g)

  return BasicILMCache{N,scaling,typeof(gdata_cache),ND,typeof(bl),typeof(nrm),typeof(a),typeof(regop),
                       typeof(Rsn),typeof(Esn),typeof(R),typeof(E),typeof(Rcurl),typeof(Ecurl),typeof(Rdiv),typeof(Ediv),typeof(L),
                       typeof(gsnorm_cache),typeof(gcurl_cache),typeof(gdiv_cache),typeof(snorm_cache),typeof(sdata_cache),typeof(sscalar_cache)}(
                       g,bl,X,nrm,a,regop,Rsn,Esn,R,E,Rcurl,Ecurl,Rdiv,Ediv,L,
                       _similar(gsnorm_cache),_similar(gsnorm_cache),
                       _similar(gcurl_cache),_similar(gdiv_cache),_similar(gdata_cache),
                       xg,yg,
                       _similar(snorm_cache),_similar(snorm_cache),_similar(sdata_cache),_similar(sscalar_cache))

end

function _surfacecache(bl::BodyList,X::VectorData{N},a,nrm,g::PhysicalGrid{ND},ddftype,scaling,
    sdata_cache,snorm_cache,sscalar_cache,gsnorm_cache,gcurl_cache,gdiv_cache,gdata_cache,L;dtype=Float64) where {N,ND}


    regop = _get_regularization(X,a,g,ddftype,scaling)
    Rsn = _regularization_matrix(regop,snorm_cache,gsnorm_cache)
    Esn = _interpolation_matrix(regop, gsnorm_cache, snorm_cache)

    R = _regularization_matrix(regop,sdata_cache,gdata_cache )
    E = _interpolation_matrix(regop, gdata_cache,sdata_cache)

    Rcurl = _regularization_matrix(regop,sscalar_cache,gcurl_cache )
    Ecurl = _interpolation_matrix(regop, gcurl_cache,sscalar_cache)

    Rdiv = _regularization_matrix(regop,sscalar_cache,gdiv_cache )
    Ediv = _interpolation_matrix(regop, gdiv_cache,sscalar_cache)

    return BasicILMCache{N,scaling,typeof(gdata_cache),ND,typeof(bl),typeof(nrm),typeof(a),typeof(regop),
        typeof(Rsn),typeof(Esn),typeof(R),typeof(E),typeof(Rcurl),typeof(Ecurl),typeof(Rdiv),typeof(Ediv),typeof(L),
        typeof(gsnorm_cache),typeof(gcurl_cache),typeof(gdiv_cache),typeof(snorm_cache),typeof(sdata_cache),typeof(sscalar_cache)}(
        g,bl,nrm,a,regop,Rsn,Esn,R,E,Rcurl,Ecurl,Rdiv,Ediv,L,
        _similar(gsnorm_cache),_similar(gsnorm_cache),
        _similar(gcurl_cache),_similar(gdiv_cache),_similar(gdata_cache),
        _similar(snorm_cache),_similar(snorm_cache),_similar(sdata_cache),_similar(sscalar_cache))

end



#=
Point collection cache
=#

struct PointCollectionCache{N,GCT,ND,PT,REGT<:Regularize,SST} <: AbstractBasicCache{N,GCT}
    pts :: PT
    g :: PhysicalGrid{ND}
    regop :: REGT
    gdata_cache :: GCT
    sdata_cache :: SST
end

function ScalarPointCollectionCache(X::VectorData{N},g::PhysicalGrid{ND};ddftype = DEFAULT_DDF,dtype = DEFAULT_DATA_TYPE) where {N,ND}
    gdata_cache = Nodes(Primal,size(g),dtype = dtype)
    sdata_cache = ScalarData(X,dtype = dtype)
    return _pointcollectioncache(X,g,gdata_cache,sdata_cache,ddftype)
end

function VectorPointCollectionCache(X::VectorData{N},g::PhysicalGrid{ND};ddftype = DEFAULT_DDF,dtype = DEFAULT_DATA_TYPE) where {N,ND}
    gdata_cache = Edges(Primal,size(g),dtype = dtype)
    sdata_cache = VectorData(X,dtype = dtype)
    return _pointcollectioncache(X,g,gdata_cache,sdata_cache,ddftype)
end

function _pointcollectioncache(X::VectorData{N},g::PhysicalGrid{ND},gdata_cache::GCT,sdata_cache::SDT,ddftype) where {N,ND,GCT,SDT}
  regop = _get_regularization(X,g,ddftype)
  return PointCollectionCache{N,GCT,ND,typeof(X),typeof(regop),SDT}(X,g,regop,gdata_cache,sdata_cache)
end

#=
Convenience functions
=#

@inline get_grid(s::AbstractBasicCache) = s.g
@inline get_bodies(s::AbstractBasicCache) = s.bl
@inline CartesianGrids.cellsize(s::AbstractBasicCache) = cellsize(s.g)
@inline Base.length(s::AbstractBasicCache{N}) where {N} = N
Base.eltype(s::AbstractBasicCache) = eltype(s.gdata_cache)

# Standardize the regularization
for ddf in [:Yang3,:Roma,:Goza,:Witchhat,:M3,:M4prime]
  @eval _get_regularization(X::VectorData{N},a::ScalarData{N},g::PhysicalGrid,::Type{CartesianGrids.$ddf},::Type{GridScaling};filter=false) where {N} =
     Regularize(X,cellsize(g),I0=origin(g),weights=a.data,ddftype=CartesianGrids.$ddf,filter=filter)

  @eval _get_regularization(X::VectorData{N},a::ScalarData{N},g::PhysicalGrid,::Type{CartesianGrids.$ddf},::Type{IndexScaling};filter=false) where {N} =
     Regularize(X,cellsize(g),I0=origin(g),issymmetric=true,ddftype=CartesianGrids.$ddf,filter=filter)

  @eval _get_regularization(X::VectorData{N},g::PhysicalGrid,::Type{CartesianGrids.$ddf}) where {N} =
     Regularize(X,cellsize(g),I0=origin(g),ddftype=CartesianGrids.$ddf,filter=false)
end

_get_regularization(body::Union{Body,BodyList},args...;kwargs...) = _get_regularization(VectorData(collect(body)),areas(body),args...;kwargs...)



# Standardize the Laplacian
_get_laplacian(coeff_factor::Real,g::PhysicalGrid,with_inverse,::Type{IndexScaling};dtype=Float64) =
               CartesianGrids.plan_laplacian(g,with_inverse=with_inverse,factor=coeff_factor,dtype=dtype)
_get_laplacian(coeff_factor::Real,g::PhysicalGrid,with_inverse,::Type{GridScaling};dtype=Float64) =
               CartesianGrids.plan_laplacian(g,with_inverse=with_inverse,factor=coeff_factor/cellsize(g)^2,dtype=dtype)

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

@inline _regularization_matrix(regop::Regularize,src::PointData,::Nothing) = nothing

@inline _regularization_matrix(regop::Regularize,::Nothing,trg) = nothing


@inline _interpolation_matrix(regop::Regularize,src::GridData,trg::PointData) =
        InterpolationMatrix(regop,src,trg)

@inline _interpolation_matrix(regop::Regularize,::Nothing,trg::PointData) = nothing

@inline _interpolation_matrix(regop::Regularize,src,::Nothing) = nothing


# APIs to generate regularization, interpolation matrices and Laplacians not generated
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

"""
    Laplacian(cache::BasicILMCache,coeff_factor::Real[,with_inverse=true])

Create an invertible Laplacian operator for the grid in `cache`,
using the index or grid scaling associated with `cache`. The operator is pre-multiplied
by the factor `coeff_factor`.
"""
Laplacian(cache::BasicILMCache{N,SCA},coeff_factor; with_inverse=true, dtype=Float64) where {N,SCA} =
    _get_laplacian(coeff_factor,cache.g,with_inverse,SCA;dtype=dtype)


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
@inline similar_grid(cache::AbstractBasicCache,kwargs...) = _similar(cache.gdata_cache,kwargs...)

"""
    similar_gridgrad(::BasicILMCache)

Get a `similar` copy of the gradient of the grid data in the cache.
"""
@inline similar_gridgrad(cache::BasicILMCache,kwargs...) = _similar(cache.gsnorm_cache,kwargs...)


"""
    similar_gridcurl(::BasicILMCache)

Get a `similar` copy of the grid curl field data in the cache.
"""
@inline similar_gridcurl(cache::BasicILMCache,kwargs...) = _similar(cache.gcurl_cache,kwargs...)

"""
    similar_griddiv(::BasicILMCache)

Get a `similar` copy of the grid div field data in the cache.
"""
@inline similar_griddiv(cache::BasicILMCache,kwargs...) = _similar(cache.gdiv_cache,kwargs...)


"""
    similar_gridgradcurl(::BasicILMCache)

Get a `similar` copy of the grid gradient-of-curl field data in the cache.
"""
@inline similar_gridgradcurl(cache::BasicILMCache;element_type=eltype(cache)) = Edges(Dual,cache.gdata_cache,dtype=element_type)


"""
    similar_surface(::BasicILMCache)

Get a `similar` copy of the basic surface point data in the cache.
"""
@inline similar_surface(cache::AbstractBasicCache,kwargs...) = _similar(cache.sdata_cache,kwargs...)

"""
    similar_surfacescalar(::BasicILMCache)

Get a `similar` copy of the surface scalar point data in the cache. This
is only used for vector-type caches. Otherwise, [`similar_surface`](@ref) should
be used.
"""
@inline similar_surfacescalar(cache::AbstractBasicCache,kwargs...) = _similar(cache.sscalar_cache,kwargs...)


_similar(::Nothing;kwargs...) = nothing
_similar(a;kwargs...) = similar(a;kwargs...)

"""
    zeros_grid(::BasicILMCache)

Get an instance of the basic grid data in the cache, with values set to zero.
"""
@inline zeros_grid(cache::AbstractBasicCache,kwargs...) = _zero(cache.gdata_cache,kwargs...)

"""
    zeros_gridgrad(::BasicILMCache)

Get an instance of the gradient of the grid data in the cache, with values set to zero.
"""
@inline zeros_gridgrad(cache::BasicILMCache,kwargs...) = _zero(cache.gsnorm_cache,kwargs...)


"""
    zeros_gridcurl(::BasicILMCache)

Get an instance of the grid curl field data in the cache, with values set to zero.
"""
@inline zeros_gridcurl(cache::BasicILMCache,kwargs...) = _zero(cache.gcurl_cache,kwargs...)

"""
    zeros_griddiv(::BasicILMCache)

Get an instance of the grid div field data in the cache, with values set to zero.
"""
@inline zeros_griddiv(cache::BasicILMCache,kwargs...) = _zero(cache.gdiv_cache,kwargs...)

"""
    zeros_gridgradcurl(::BasicILMCache)

Get an instance of the grid gradient-of-curl field data in the cache, with values set to zero.
"""
@inline zeros_gridgradcurl(cache::BasicILMCache;element_type=eltype(cache)) = Edges(Dual,cache.gdata_cache,dtype=element_type)


"""
    zeros_surface(::BasicILMCache)

Get an instance of the basic surface point data in the cache, with values set to zero.
"""
@inline zeros_surface(cache::AbstractBasicCache,kwargs...) = _zero(cache.sdata_cache,kwargs...)

"""
    zeros_surfacescalar(::BasicILMCache)

Get an instance of the surface scalar point data in the cache, with values set to zero.
This is only used for vector-type caches. Otherwise, [`zeros_surface`](@ref) should
be used.
"""
@inline zeros_surfacescalar(cache::AbstractBasicCache,kwargs...) = _zero(cache.sscalar_cache,kwargs...)


_zero(a;kwargs...) = zero(a;kwargs...)
_zero(::Nothing;kwargs...) = nothing

"""
    ones_grid(::BasicILMCache)

Get an instance of the basic grid data in the cache, with values set to unity.
"""
@inline ones_grid(cache::AbstractBasicCache;kwargs...) = ones(cache.gdata_cache;kwargs...)

"""
    ones_grid(::BasicILMCache,dim)

For a vector-type cache, get an instance of the basic grid data in the cache, with values set to unity
in dimension `dim`.
"""
@inline ones_grid(cache::AbstractBasicCache{N,GCT},dim;kwargs...) where {N,GCT<:Edges} = ones(cache.gdata_cache,dim;kwargs...)


"""
    ones_gridgrad(::BasicILMCache)

Get an instance of the gradient of the grid data in the cache,
with values set to unity.
"""
@inline ones_gridgrad(cache::BasicILMCache;kwargs...) = ones(cache.gsnorm_cache;kwargs...)


"""
    ones_gridgrad(::BasicILMCache,dim)

Get an instance of the gradient of the grid data in the cache, in direction `dim`,
with values set to unity. If the data are of type `TensorGridData`, then
`dim` takes values from 1 to 2^2.
"""
@inline ones_gridgrad(cache::BasicILMCache,dim;kwargs...) = ones(cache.gsnorm_cache,dim;kwargs...)


"""
    ones_gridcurl(::BasicILMCache)

Get an instance of the grid curl field data in the cache, with values set to unity.
"""
@inline ones_gridcurl(cache::BasicILMCache,kwargs...) = _ones(cache.gcurl_cache,kwargs...)

"""
    ones_griddiv(::BasicILMCache)

Get an instance of the grid div field data in the cache, with values set to unity.
"""
@inline ones_griddiv(cache::BasicILMCache,kwargs...) = _ones(cache.gdiv_cache,kwargs...)


"""
    ones_gridgradcurl(::BasicILMCache)

Get an instance of the grid gradient-of-curl field data in the cache, with values set to unity.
"""
@inline ones_gridgradcurl(cache::BasicILMCache;element_type=eltype(cache)) =
      (d = Edges(Dual,cache.gdata_cache,dtype=element_type); fill!(d,one(element_type)))


"""
    ones_surface(::BasicILMCache)

Get an instance of the basic surface point data in the cache, with values set to unity.
"""
@inline ones_surface(cache::AbstractBasicCache;kwargs...) = ones(cache.sdata_cache;kwargs...)

"""
    ones_surface(::BasicILMCache,dim)

Get an instance of the basic surface point data in the cache, with values set to unity
in dimension `dim`. This only works for a vector-type cache.
"""
@inline ones_surface(cache::AbstractBasicCache{N,GCT},dim;kwargs...) where {N,GCT<:Edges} = ones(cache.sdata_cache,dim;kwargs...)


"""
    ones_surfacescalar(::BasicILMCache)

Get an instance of the surface scalar point data in the cache, with values set to unity.
This is only used for vector-type caches. Otherwise, [`ones_surface`](@ref) should
be used.
"""
@inline ones_surfacescalar(cache::AbstractBasicCache;kwargs...) = _ones(cache.sscalar_cache;kwargs...)


_ones(a...;kwargs...) = ones(a...;kwargs...)
_ones(::Nothing;kwargs...) = nothing


"""
    x_grid(::BasicILMCache)

Return basic grid data filled with the grid `x` coordinate.
If the grid data is scalar, then the output of this function is
of the same type. If the grid
data is vector, then the output is of type `Edges{Primal}`,
and the coordinates of each component are in `u`, `v` fields.
"""
x_grid(cache::BasicILMCache)  = cache.xg

#=
function x_grid(cache::BasicILMCache{N,SCA,GT}) where {N,SCA,GT<:Edges{Primal}}
    p = zeros_grid(cache)
    xu, _ = coordinates(p.u,cache.g)
    xv, _ = coordinates(p.v,cache.g)
    p.u .= xu
    p.v .= xv
    return p
end
=#

function _x_grid(gdata::GT,g::PhysicalGrid) where {GT<:Nodes{Primal}}
    xc, _ = coordinates(gdata,g)
    p = _zero(gdata)
    p .= xc
    return p
end

function _x_grid(gdata::GT,g::PhysicalGrid) where {GT<:Edges{Primal}}
    xu, _ = coordinates(gdata.u,g)
    xv, _ = coordinates(gdata.v,g)
    p = _zero(gdata)
    p.u .= xu
    p.v .= xv
    return p
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
    x_griddiv(::BasicILMCache)

Return basic grid div field data filled with the grid `x` coordinate
"""
@inline x_griddiv(cache::BasicILMCache) = _x_griddiv(cache,Val(cache_datatype(cache)))

_x_griddiv(cache,::Val{:scalar}) = nothing

function _x_griddiv(cache,::Val{:vector})
    xc, _ = coordinates(cache.gdiv_cache,cache.g)
    p = zeros_griddiv(cache)
    p .= xc
end

"""
    x_gridgrad(::BasicILMCache)

Return basic grid gradient field data filled with the grid `x` coordinate.
If the grid data is scalar, then the output of this function is
of `Edges{Primal}` type and the coordinate field for each component
can be accessed with the `u` and `v` field, respectively. If the grid
data is vector, then the output is of type `EdgeGradient{Primal}`,
and the coordinates are in `dudx`, `dudy`, `dvdx`, and `dvdy` fields.
"""
function x_gridgrad(cache::BasicILMCache{N,SCA,GT}) where {N,SCA,GT<:Nodes{Primal}}
    p = zeros_gridgrad(cache)
    xu, _ = coordinates(p.u,cache.g)
    xv, _ = coordinates(p.v,cache.g)
    p.u .= xu
    p.v .= xv
    return p
end

function x_gridgrad(cache::BasicILMCache{N,SCA,GT}) where {N,SCA,GT<:Edges{Primal}}
    p = zeros_gridgrad(cache)
    xdudx, _ = coordinates(p.dudx,cache.g)
    xdudy, _ = coordinates(p.dudy,cache.g)
    p.dudx .= xdudx
    p.dudy .= xdudy
    p.dvdx .= xdudy
    p.dvdy .= xdudx
    return p
end

"""
    y_grid(::BasicILMCache)

Return basic grid data filled with the grid `y` coordinate.
If the grid data is scalar, then the output of this function is
of the same type. If the grid
data is vector, then the output is of type `Edges{Primal}`,
and the coordinates of each component are in `u`, `v` fields.
"""
y_grid(cache::BasicILMCache) = cache.yg

function _y_grid(gdata::GT,g::PhysicalGrid) where {GT<:Nodes{Primal}}
    _, yc  = coordinates(gdata,g)
    p = _zero(gdata)
    p .= yc'
    return p
end

function _y_grid(gdata::GT,g::PhysicalGrid) where {GT<:Edges{Primal}}
    _, yu  = coordinates(gdata.u,g)
    _, yv  = coordinates(gdata.v,g)
    p = _zero(gdata)

    p.u .= yu'
    p.v .= yv'
    return p
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

"""
    y_griddiv(::BasicILMCache)

Return basic grid div field data filled with the grid `y` coordinate
"""
@inline y_griddiv(cache::BasicILMCache) = _y_griddiv(cache,Val(cache_datatype(cache)))

_y_griddiv(cache,::Val{:scalar}) = nothing

function _y_griddiv(cache,::Val{:vector})
    _,yc = coordinates(cache.gdiv_cache,cache.g)
    p = zeros_griddiv(cache)
    p .= yc'
end

"""
    y_gridgrad(::BasicILMCache)

Return basic grid gradient field data filled with the grid `y` coordinate.
If the grid data is scalar, then the output of this function is
of `Edges{Primal}` type and the coordinate field for each component
can be accessed with the `u` and `v` field, respectively. If the grid
data is vector, then the output is of type `EdgeGradient{Primal}`,
and the coordinates are in `dudx`, `dudy`, `dvdx`, and `dvdy` fields.
"""
function y_gridgrad(cache::BasicILMCache{N,SCA,GT}) where {N,SCA,GT<:Nodes{Primal}}
    p = zeros_gridgrad(cache)
    _, yu = coordinates(p.u,cache.g)
    _, yv = coordinates(p.v,cache.g)
    p.u .= yu'
    p.v .= yv'
    return p
end

function y_gridgrad(cache::BasicILMCache{N,SCA,GT}) where {N,SCA,GT<:Edges{Primal}}
    p = zeros_gridgrad(cache)
    _, ydudx = coordinates(p.dudx,cache.g)
    _, ydudy = coordinates(p.dudy,cache.g)
    p.dudx .= ydudx'
    p.dudy .= ydudy'
    p.dvdx .= ydudy'
    p.dvdy .= ydudx'
    return p
end


# Extend operators on body points
"""
    points(cache::BasicILMCache)

Return the coordinates (as `VectorData`) of the surface points associated with `cache`
"""
points(cache::BasicILMCache) = cache.pts

"""
    points(cache::PointCollectionCache)

Return the coordinates (as `VectorData`) of the surface points associated with `cache`
"""
points(cache::PointCollectionCache) = cache.pts

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
    integrate(u::GridData,cache::BasicILMCache)

Calculate the discrete volume integral of grid data `u`.
If `u` is `VectorGridData`, then this returns a vector of the integrals in
each coordinate direction. This integral produces the same result regardless
if `GridScaling` or `IndexScaling` is used.
"""
@inline integrate(u::GridData,cache::BasicILMCache) = integrate(u,cache.g)



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


"""
    arcs(cache::BasicILMCache)

Return `ScalarData` of arc length coordinates for the body surface(s) in
the given cache `cache`.
"""
arcs(cache::BasicILMCache) = arcs(cache.bl)
