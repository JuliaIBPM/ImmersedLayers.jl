# Forcing classes (areas, lines, points)

#=
Notes:
- The principle here is that the problem specification will supply the shape
 of the forcing region (area, line) and a model function to apply this forcing.
- The forcing model function should be able to make use of other system data, like physical
  parameters
=#

abstract type AbstractForcingModel end
abstract type AbstractRegionCache end

for ftype in [:Area,:Line,:Point]
    fmname = Symbol(string(ftype)*"ForcingModel")
    typename = lowercase(string(ftype))
    fmcache = string(ftype)*"RegionCache"

    eval(quote
      export $fmname

      struct $fmname{RT,TT,MDT<:Function} <: AbstractForcingModel
          shape :: RT
          transform :: TT
          kwargs :: AbstractDict
          fcn :: MDT
          $fmname(shape,transform,fcn;kwargs...) = 
                    new{typeof(shape),typeof(transform),typeof(fcn)}(shape,transform,kwargs,fcn)
          $fmname(shape,fcn;kwargs...) = 
                    new{typeof(shape),MotionTransform{2},typeof(fcn)}(shape,MotionTransform{2}(),kwargs,fcn)          
      end

    end)
end

"""
      AreaForcingModel(shape::Union{Body,BodyList},transform::MotionTransform,model_function!)

  Bundles a `shape` (i.e., a `Body`, `BodyList`,), a `transform` (to specify where to
  place the shape), and a `model_function!` (a function
  that returns the strength of the forcing) for area-type forcing.
  `model_function!` must be in-place with a signature of the form

      model_function!(str,state,t,fcache,phys_params)

  where `str` is the strength to be returned, `state` the state vector,
  `t` is time, `fcache` is a corresponding `AreaRegionCache`
  and `phys_params` are user-supplied physical parameters. Any of these can
  be utilized to compute the strength.

  There are a number of keyword arguments that can be passed in:
  `ddftype =`,specifying the type of DDF; `spatialfield =` to provide
  an `AbstractSpatialField` to help in evaluating the strength.
  (The resulting field is available to `model_function!` in the `generated_field` field
    of `fcache`.)
""" AreaForcingModel(::Union{Body,BodyList},::MotionTransform,::Function)

"""
    AreaForcingModel(model_function!)

  Creates area-type forcing over the entire domain, using a `model_function!` (a function
  that returns the strength of the forcing).

  `model_function!` must be in-place with a signature of the form

      model_function!(str,state,t,fcache,phys_params)

  where `str` is the strength to be returned, `state` the state vector,
  `t` is time, `fcache` is a corresponding `AreaRegionCache`
  and `phys_params` are user-supplied physical parameters. Any of these can
  be utilized to compute the strength.

  The keyword `spatialfield =` can provide
  an `AbstractSpatialField` to help in evaluating the strength.
  (The resulting field is available to `model_function!` in the `generated_field` field
    of `fcache`.)
"""
AreaForcingModel(fcn::Function;kwargs...) = AreaForcingModel(nothing,nothing,fcn;kwargs...)

"""
      LineForcingModel(shape::Union{Body,BodyList},transform::MotionTransform,model_function!)

  Bundles a `shape` (i.e., a `Body`, `BodyList`,), a `transform` (to specify where to
  place the shape), and a `model_function!` (a function
  that returns the strength of the forcing) for line-type forcing.
  `model_function!` must be in-place with a signature of the form

      model_function!(str,state,t,fcache,phys_params)

  where `str` is the strength to be returned, `state` the state vector,
  `t` is time, `fcache` is a corresponding `LineRegionCache`
  and `phys_params` are user-supplied physical parameters. Any of these can
  be utilized to compute the strength.

  The keyword `ddftype =` can be used to specify the type of DDF.
""" LineForcingModel(::Union{Body,BodyList},::MotionTransform,::Function)

"""
      PointForcingModel(pts::VectorData,transform::MotionTransform,model_function!)

  Bundles point coordinates `pts`, a `transform` (to specify where the origin of the points'
  coordinate system is relative to the inertial system's origin), and a `model_function!` (a function
  that returns the strength of the forcing) for point-type forcing.
  `model_function!` must be in-place with a signature of the form

      model_function!(str,state,t,fcache,phys_params)

  where `str` is the strength to be returned, `state` the state vector,
  `t` is time, `fcache` is a corresponding `PointRegionCache`
  and `phys_params` are user-supplied physical parameters. Any of these can
  be utilized to compute the strength.

  The keyword `ddftype =` can be used to specify the type of DDF.
""" PointForcingModel(::VectorData,::MotionTransform,::Function)

"""
    PointForcingModel(point_function::Function,transform::MotionTransform,model_function!::Function)

Bundles a `point_function` (a function that returns the positions of forcing points), a `transform`
(to specify where the origin of the points' coordinate system is relative to the inertial system's origin),
and a `model_function` (a function that returns the strength of the forcing) for point-type forcing.

  `point_function` must have an out-of-place signature of the form

      point_function(state,t,fcache,phys_params)

  where `state` the state vector, `t` is time, `fcache` is a corresponding `PointRegionCache`
  and `phys_params` are user-supplied physical parameters. Any of these can
  be utilized to compute the positions. It must return either a vector for
  each coordinate or `VectorData`.

  `model_function!` must have an in-place signature of the form

      model_function!(str,state,t,fcache,phys_params)

  where `str` is the strength to be returned.
""" PointForcingModel(::Function,::MotionTransform,::Function)


#=
Region caches
=#

"""
    AreaRegionCache(shape::Body/BodyList,cache::BasicILMCache)

Create an area region (basically, a mask) of the shape(s) `shape`, using
the data in `cache` to provide the details of the grid and regularization.
""" AreaRegionCache(::Union{Body,BodyList},::BasicILMCache)

"""
    AreaRegionCache(cache::BasicILMCache)

Create an area region that spans the entire domain, using
the data in `cache` to provide the grid details.
""" AreaRegionCache(::BasicILMCache)


"""
    LineRegionCache(shape::Body/BodyList,cache::BasicILMCache)

Create a line region of the shape(s) `shape`, using
the data in `cache` to provide the details of the regularization.
""" LineRegionCache(::Union{Body,BodyList},::BasicILMCache)

"""
    PointRegionCache(pts::VectorData,cache::BasicILMCache[,kwargs])

Create a regularized point collection based on points `pts`, using the data
in `cache` to provide the details of the regularization. The `kwargs` can be used to override
the regularization choices, such as ddf.
""" PointRegionCache(::VectorData,::BasicILMCache)


struct AreaRegionCache{MT,ST,SFT,SGT,CT<:BasicILMCache} <: AbstractRegionCache
    mask :: MT
    str :: ST
    spatialfield :: SFT
    generated_field :: SGT
    cache :: CT
end

struct LineRegionCache{ACT,ST,CT<:BasicILMCache} <: AbstractRegionCache
    s :: ACT
    str :: ST
    cache :: CT
end

struct PointRegionCache{ST,CT<:PointCollectionCache} <: AbstractRegionCache
    str :: ST
    cache :: CT
end



# Some constructors of the region caches that distinguish scalar and vector data
for f in [:Scalar,:Vector]
    cname = Symbol("Surface"*string(f)*"Cache")
    pcname = Symbol(string(f)*"PointCollectionCache")
    gdtype = Symbol(string(f)*"GridData")
    @eval function _arearegioncache(g::PhysicalGrid,shape::BodyList,data_prototype::$gdtype,L::Laplacian;spatialfield=nothing,Xi_to_ref=MotionTransform{2}(),kwargs...)
        cache = $cname(shape,g;L=L,kwargs...)
        m = mask(cache)
        str = similar_grid(cache)

        # Generate spatial field with given transform
        sf_updated = _update_spatialfield(spatialfield,Xi_to_ref)
        gf = _generatedfield(str,sf_updated,g)

        return AreaRegionCache{typeof(m),typeof(str),typeof(spatialfield),typeof(gf),typeof(cache)}(m,str,spatialfield,gf,cache)
    end

    @eval function _arearegioncache(g::PhysicalGrid,data_prototype::$gdtype,L::Laplacian;spatialfield=nothing,Xi_to_ref=MotionTransform{2}(),kwargs...)
        cache = $cname(g;L=L,kwargs...)
        m = mask(cache)
        str = similar_grid(cache)

        # Generate spatial field with given transform
        sf_updated = _update_spatialfield(spatialfield,Xi_to_ref)
        gf = _generatedfield(str,sf_updated,g)

        return AreaRegionCache{typeof(m),typeof(str),typeof(spatialfield),typeof(gf),typeof(cache)}(m,str,spatialfield,gf,cache)
    end

    @eval function _lineregioncache(g::PhysicalGrid,shape::BodyList,data_prototype::$gdtype,L::Laplacian;Xi_to_ref=MotionTransform{2}(),kwargs...)
        cache = $cname(shape,g;L=L,kwargs...)
        pts = points(shape)
        s = arcs(shape)
        str = similar_surface(cache)
        return LineRegionCache{typeof(s),typeof(str),typeof(cache)}(s,str,cache)
    end

    @eval function _pointregioncache(g::PhysicalGrid,pts::VectorData,data_prototype::$gdtype;Xi_to_ref=MotionTransform{2}(),kwargs...)
        cache = $pcname(pts,g;kwargs...)

        str = similar(cache.sdata_cache)
        return PointRegionCache{typeof(str),typeof(cache)}(str,cache)
    end

    @eval function _pointregioncache(g::PhysicalGrid,pts::Function,data_prototype::$gdtype;Xi_to_ref=MotionTransform{2}(),kwargs...)
        cache = $pcname(VectorData(0),g;kwargs...)

        str = similar(cache.sdata_cache)
        return PointRegionCache{typeof(str),typeof(cache)}(str,cache)
    end

end

for f in [:AreaRegionCache,:LineRegionCache]
    underscore_f = Symbol("_"*lowercase(string(f)))
    @eval $f(shape::Union{Body,BodyList},cache::BasicILMCache{N,SCA};scaling=SCA,kwargs...) where {N,SCA} = $underscore_f(cache.g,shape,similar_grid(cache),cache.L;scaling=scaling,kwargs...)
    @eval $underscore_f(g::PhysicalGrid,shape::Body,a...;kwargs...) = $underscore_f(g,BodyList([shape]),a...;kwargs...)
end

AreaRegionCache(cache::AbstractBasicCache;kwargs...) =
      _arearegioncache(cache.g,similar_grid(cache),cache.L;kwargs...)

AreaRegionCache(::Any,cache::AbstractBasicCache;kwargs...) =
      AreaRegionCache(cache;kwargs...)


PointRegionCache(pts::Union{VectorData,Function},cache::AbstractBasicCache;kwargs...) =
      _pointregioncache(cache.g,pts,similar_grid(cache);kwargs...)


 
_update_spatialfield(::Nothing,Xi_to_ref) = nothing
_update_spatialfield(sf::AbstractSpatialField,Xi_to_ref) = sf

function _update_spatialfield(sg::SpatialGaussian{C,D},Xi_to_ref) where {C,D}
    @unpack Σ, x0, A = sg

    F = eigen(Σ)
    Xi_to_g = MotionTransform(x0,F.vectors)
    Xref_to_g = Xi_to_g*inv(Xi_to_ref)
    R = rotation(Xref_to_g)
    x0 = translation(Xref_to_g)

    return SpatialGaussian(R*Diagonal(F.values)*R',x0,A;derivdir=D)
end

function _update_spatialfield(sfv::Vector{<:AbstractSpatialField},Xi_to_ref)
    R = rotation(Xi_to_ref)
    sfv_rotated = AbstractSpatialField[EmptySpatialField(), EmptySpatialField()]
    for (j,sf) in enumerate(sfv)
        sf_updated = _update_spatialfield(sf,Xi_to_ref)
        for i in eachindex(sfv_rotated)
            sfv_rotated[i] += R[i,j]*sf_updated
        end
    end
    return sfv_rotated
end 

     

_generatedfield(field_prototype::GridData,s,g::PhysicalGrid) = nothing
_generatedfield(field_prototype::GridData,s::Union{T,Vector{T}},g::PhysicalGrid) where {T<:AbstractSpatialField} =
          GeneratedField(field_prototype,s,g)


"""
    mask(ar::AreaRegionCache)

Return the mask for the given area region `ar`.
"""
mask(ar::AreaRegionCache) = ar.mask

"""
    arcs(lr::LineRegionCache)

Return the vector of arc length coordinates for the given line region `lr`.
"""
arcs(lr::LineRegionCache) = lr.s

"""
    points(pr::PointRegionCache)

Return the vector of coordinates of points associated with `pr`.
"""
points(pr::PointRegionCache) = points(pr.cache)

#=
Assembly of the forcing model and forcing region
=#

"""
    ForcingModelAndRegion

A type that holds the forcing model function, region, and cache
# Constructors

`ForcingModelAndRegion(model::AbstractForcingModel,cache::BasicILMCache)`

`ForcingModelAndRegion(modellist::Vector{AbstractForcingModel},cache::BasicILMCache)`

These forms generally get called when building the extra cache. They
also gracefully generate an empty list of models if one passes along `nothing`
in the first argument.
""" ForcingModelAndRegion(::AbstractForcingModel,::BasicILMCache)

struct ForcingModelAndRegion{RT<:AbstractRegionCache,ST,TT,MT,KT}
    region_cache :: RT
    shape :: ST
    transform :: TT
    fcn :: MT
    kwargs :: KT
end

# This function declaration prevents errors with unknown TypeVar  
function _forcingmodelandregion(::Union{AbstractForcingModel,ForcingModelAndRegion},::MotionTransform,::BasicILMCache) end

for f in [:Area,:Line,:Point]
  modtype = Symbol(string(f)*"ForcingModel")
  regcache = Symbol(string(f)*"RegionCache")
  @eval function _forcingmodelandregion(model::Union{$modtype,ForcingModelAndRegion{T}},Xi_to_ref::MotionTransform,cache::BasicILMCache) where {T<:$regcache}

    # Place the shape at the desired position
    shape = deepcopy(model.shape)
    _update_shape!(shape,model.transform,Xi_to_ref)

    region_cache = $regcache(shape,cache;Xi_to_ref=Xi_to_ref,model.kwargs...)
    ForcingModelAndRegion{typeof(region_cache),typeof(model.shape),typeof(model.transform),typeof(model.fcn),typeof(model.kwargs)}(region_cache,model.shape,model.transform,model.fcn,model.kwargs)
  end
  
  #@eval ForcingModelAndRegion(flist::Vector{T},cache::BasicILMCache;kwargs...) where {T<:$modtype} = 
  #          ForcingModelAndRegion(convert(Vector{AbstractForcingModel},flist),cache;kwargs...)

end

_update_shape!(shape,::Nothing,Xi_to_ref) = nothing
_update_shape!(shape::Body,transform::MotionTransform,Xi_to_ref::MotionTransform) = update_body!(shape,transform*inv(Xi_to_ref))
_update_shape!(shape::Function,transform::MotionTransform,Xi_to_ref::MotionTransform) = nothing

function _update_shape!(pts::VectorData,transform::MotionTransform,Xi_to_ref::MotionTransform)
    full_transform = transform*inv(Xi_to_ref)
    u, v = full_transform(pts.u,pts.v)
    pts.u .= u
    pts.v .= v
    nothing
end

ForcingModelAndRegion(f::AbstractForcingModel,cache::BasicILMCache;kwargs...) = ForcingModelAndRegion(AbstractForcingModel[f],cache;kwargs...)


function ForcingModelAndRegion(flist::Vector{T},cache::BasicILMCache;Xi_to_ref = MotionTransform{2}()) where {T<: Union{AbstractForcingModel,ForcingModelAndRegion}}
   fmlist = ForcingModelAndRegion[]
   for f in flist
     push!(fmlist,_forcingmodelandregion(f,Xi_to_ref,cache))
   end
   return fmlist
end

ForcingModelAndRegion(flist,cache::BasicILMCache;kwargs...) = ForcingModelAndRegion(AbstractForcingModel[],cache;kwargs...)

#=
Application of forcing
=#


"""
    apply_forcing!(out,y,x,t,fv::Vector{ForcingModelAndRegion},sys::ILMSystem)

Return the total contribution of forcing in the vector `fv` to `out`,
based on the current state `y`, auxiliary state `x`, time `t`, and ILM system `sys`.
Note that `out` is zeroed before the contributions are added.
"""
apply_forcing!(out,y,x,t,fr::Vector{<:ForcingModelAndRegion},sys::ILMSystem) = 
            apply_forcing!(out,y,x,t,fr,sys.phys_params,sys.motions,sys.base_cache)

function apply_forcing!(out,y,x,t,fr::Vector{<:ForcingModelAndRegion},phys_params,motions,base_cache::BasicILMCache)

    # In case there is motion and the motion references axes other than the inertial ones
    # then regenerate the forcing cache. Otherwise just use the given one.
    fr_updated, Xi_to_ref = _regenerate_forcing_cache(fr,x,motions,base_cache)
    fill!(out,0.0)
    for f in fr_updated
        _apply_forcing!(out,y,t,f,phys_params,Xi_to_ref)
    end
    return out
end


_regenerate_forcing_cache(fr,x,::Nothing,cache) = fr, MotionTransform{2}()

function _regenerate_forcing_cache(fr,x,motions::ILMMotion,cache)
    @unpack reference_body, m = motions
    _regenerate_forcing_cache(fr,x,m,cache,Val(reference_body))
end

_regenerate_forcing_cache(fr,x,::Nothing,cache,::Val{N}) where {N} = fr, MotionTransform{2}()

_regenerate_forcing_cache(fr,x,m::RigidBodyMotion,cache,::Val{0}) = fr, MotionTransform{2}()

function _regenerate_forcing_cache(fr,x,m::RigidBodyMotion,cache,::Val{N}) where {N}
    Xl = body_transforms(x,m)
    Xi_to_ref = Xl[N]
    ForcingModelAndRegion(fr,cache;Xi_to_ref=Xi_to_ref), Xi_to_ref
end




"""
    apply_forcing!(dy,y,x,t,f::ForcingModelAndRegion,sys::ILMSystem)

Return the contribution of forcing in `f` to the right-hand side `dy`
based on the current state `y`, auxiliary state `x`, time `t`, and ILM system `sys`.
"""
apply_forcing!(dy,y,x,t,fr::ForcingModelAndRegion,a...) = apply_forcing!(dy,y,x,t,[fr],a...)

#=
The following define how forcing of each type get applied. Each one
calls the forcing model to determine the strength.
=#

function _apply_forcing!(dy,y,t,fcache::ForcingModelAndRegion{<:AreaRegionCache},phys_params,Xi_to_ref)
    @unpack region_cache, fcn = fcache
    @unpack str = region_cache

    fill!(str,0.0)
    fcn(str,y,t,region_cache,phys_params)

    dy .+= str.*mask(region_cache)
    return dy
end

function _apply_forcing!(dy,y,t,fcache::ForcingModelAndRegion{<:LineRegionCache},phys_params,Xi_to_ref)
    @unpack region_cache, fcn = fcache
    @unpack str, cache = region_cache

    fill!(str,0.0)
    fcn(str,y,t,region_cache,phys_params)

    fill!(cache.gdata_cache,0.0)
    regularize!(cache.gdata_cache,str,cache)
    dy .+= cache.gdata_cache
    return dy
end

function _apply_forcing!(dy,y,t,fcache::ForcingModelAndRegion{<:PointRegionCache,T},phys_params,Xi_to_ref) where T
    @unpack region_cache, fcn = fcache
    @unpack str, cache = region_cache
    @unpack regop = cache

    fill!(str,0.0)
    fcn(str,y,t,region_cache,phys_params)

    fill!(cache.gdata_cache,0.0)
    regop(cache.gdata_cache,str)
    dy .+= cache.gdata_cache
    return dy
end

function _apply_forcing!(dy,y,t,fcache::ForcingModelAndRegion{<:PointRegionCache,<:Function},phys_params,Xi_to_ref)
    @unpack region_cache, transform, shape, fcn, kwargs = fcache
    @unpack cache = region_cache

    # `shape` is a function in this case, used to obtain the point coordinates
    # Use is to to generate an instantaneous PointRegionCache
    _pts = shape(y,t,region_cache,phys_params)
    typeof(_pts) <: Tuple ? pts = VectorData(_pts...) : pts = _pts
    _update_shape!(pts,transform,Xi_to_ref)

    new_region_cache = PointRegionCache(pts,cache;kwargs...)
    @unpack str, cache = new_region_cache
    @unpack regop = cache

    fill!(str,0.0)
    fcn(str,y,t,region_cache,phys_params)

    fill!(cache.gdata_cache,0.0)
    regop(cache.gdata_cache,str)
    dy .+= cache.gdata_cache
    return dy
end
