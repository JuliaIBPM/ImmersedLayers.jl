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

      @doc """
            $($fmname)(shape::Union{Body,BodyList,VectorData},model_function!)

        Bundles a `shape` (i.e., a `Body`, `BodyList`, or `VectorData`) and a `model_function!` (a function
        that returns the strength of the forcing) for $($typename)-type forcing.
        `model_function!` be in-place with a signature of the form

            model_function!(str,x,t,fcache,phys_params)

        where `str` is the strength to be returned, `x` the state vector,
        `t` is time, `fcache` is a corresponding `$($fmcache)`
        and `phys_params` are user-supplied physical parameters. Any of these can
        be utilized to compute the strength.
        """
        struct $fmname{RT<:Union{Body,BodyList,VectorData,Function},KT,MDT<:Function} <: AbstractForcingModel
          shape :: RT
          kwargs :: KT
          fcn :: MDT
          $fmname(shape,fcn;kwargs...) = new{typeof(shape),typeof(kwargs),typeof(fcn)}(shape,kwargs,fcn)
        end
      end)
end

"""
    PointForcingModel(point_function::Function,model_function!::Function)

Bundles a `point_function` (a function that returns the positions of forcing points)
and a `model_function` (a function that returns the strength of the forcing) for point-type forcing.

  `point_function` must have an out-of-place signature of the form

      point_function(x,t,fcache,phys_params)

  where `x` the state vector, `t` is time, `fcache` is a corresponding `PointRegionCache`
  and `phys_params` are user-supplied physical parameters. Any of these can
  be utilized to compute the positions. It must return either `x` and `y` as
  a tuple of vectors or `VectorData`.

  `model_function!` must have an in-place signature of the form

      model_function!(str,x,t,fcache,phys_params)

  where `str` is the strength to be returned.
""" PointForcingModel(::Function,::Function)

#=
Region caches
=#

struct AreaRegionCache{MT,ST,CT<:BasicILMCache} <: AbstractRegionCache
    mask :: MT
    str :: ST
    cache :: CT
end

struct LineRegionCache{ACT,ST,CT<:BasicILMCache} <: AbstractRegionCache
    arccoord :: ACT
    str :: ST
    cache :: CT
end

struct PointRegionCache{ST,CT<:PointCollectionCache} <: AbstractRegionCache
    str :: ST
    cache :: CT
end

"""
    AreaRegionCache(shape::Body/BodyList,cache::BasicILMCache)

Create an area region (basically, a mask) of the shape(s) `shape`, using
the data in `cache` to provide the details of the regularization.
""" AreaRegionCache(::Union{Body,BodyList},::BasicILMCache)

"""
    LineRegionCache(shape::Body/BodyList,cache::BasicILMCache)

Create a line region of the shape(s) `shape`, using
the data in `cache` to provide the details of the regularization.
""" LineRegionCache(::Union{Body,BodyList},::BasicILMCache)

# Some constructors of the region caches that distinguish scalar and vector data
for f in [:Scalar,:Vector]
    cname = Symbol("Surface"*string(f)*"Cache")
    pcname = Symbol(string(f)*"PointCollectionCache")
    gdtype = Symbol(string(f)*"GridData")
    @eval function AreaRegionCache(g::PhysicalGrid,shape::BodyList,data_prototype::$gdtype;kwargs...)
        cache = $cname(shape,g;kwargs...)
        m = mask(cache)
        str = similar_grid(cache)
        return AreaRegionCache{typeof(m),typeof(str),typeof(cache)}(m,str,cache)
    end

    @eval function LineRegionCache(g::PhysicalGrid,shape::BodyList,data_prototype::$gdtype;kwargs...)
        cache = $cname(shape,g;kwargs...)
        pts = points(shape)

        # This needs to be written specially for closed vs open bodies and for lists
        arccoord = ScalarData(pts)
        arccoord .= accumulate(+,dlengthmid(shape))

        str = similar_surface(cache)
        return LineRegionCache{typeof(arccoord),typeof(str),typeof(cache)}(arccoord,str,cache)
    end

    @eval function PointRegionCache(g::PhysicalGrid,pts::VectorData,data_prototype::$gdtype;kwargs...)
        cache = $pcname(pts,g;kwargs...)

        str = similar(cache.sdata_cache)
        return PointRegionCache{typeof(str),typeof(cache)}(str,cache)
    end

    @eval function PointRegionCache(g::PhysicalGrid,pts::Function,data_prototype::$gdtype;kwargs...)
        cache = $pcname(VectorData(0),g;kwargs...)

        str = similar(cache.sdata_cache)
        return PointRegionCache{typeof(str),typeof(cache)}(str,cache)
    end

end

for f in [:AreaRegionCache,:LineRegionCache]
    @eval $f(shape::Union{Body,BodyList},cache::BasicILMCache{N,SCA};scaling=SCA,kwargs...) where {N,SCA} = $f(cache.g,shape,similar_grid(cache);scaling=scaling,kwargs...)
    @eval $f(g,shape::Body,a...;kwargs...) = $f(g,BodyList([shape]),a...;kwargs...)
end

PointRegionCache(pts::Union{VectorData,Function},cache::AbstractBasicCache;kwargs...) =
      PointRegionCache(cache.g,pts,similar_grid(cache);kwargs...)



"""
    mask(ar::AreaRegionCache)

Return the mask for the given area region `ar`.
"""
mask(ar::AreaRegionCache) = ar.mask

"""
    arccoord(lr::LineRegionCache)

Return the vector of arc length coordinates for the given line region `lr`.
"""
arccoord(lr::LineRegionCache) = lr.arccoord

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
# Constructor

`ForcingModelAndRegion(model::AbstractForcingModel,cache::BasicILMCache)`

This form gets called generally when building the extra cache.
""" ForcingModelAndRegion(::AbstractForcingModel,::BasicILMCache)

struct ForcingModelAndRegion{RT<:AbstractRegionCache,ST,MT,KT}
    region_cache :: RT
    shape :: ST
    fcn :: MT
    kwargs :: KT
end

for f in [:Area,:Line,:Point]
  modtype = Symbol(string(f)*"ForcingModel")
  regcache = Symbol(string(f)*"RegionCache")
  @eval function ForcingModelAndRegion(model::$modtype,cache::BasicILMCache)
      region_cache = $regcache(model.shape,cache;model.kwargs...)
      ForcingModelAndRegion(region_cache,model.shape,model.fcn,model.kwargs)
  end
end

function ForcingModelAndRegion(flist::Vector{<:AbstractForcingModel},cache::BasicILMCache)
   fmlist = ForcingModelAndRegion[]
   for f in flist
     push!(fmlist,ForcingModelAndRegion(f,cache))
   end
   return fmlist
end


"""
    apply_forcing!(out,x,t,fv::Vector{ForcingModelAndRegion},phys_params)

Return the total contribution of forcing in the vector `fv` to `out`,
based on the current state `x`, time `t`, and physical parameters in `phys_params`.
Note that `out` is zeroed before the contributions are added.
"""
function apply_forcing!(out,x,t,fr::Vector{<:ForcingModelAndRegion},phys_params)
    fill!(out,0.0)
    for f in fr
        _apply_forcing!(out,x,t,f,phys_params)
    end
    return out
end

"""
    apply_forcing!(dx,x,t,f::ForcingModelAndRegion,phys_params)

Return the contribution of forcing in `f` to the right-hand side `dx`
based on the current state `x`, time `t`, and physical parameters in `phys_params`.
"""
apply_forcing!(dx,x,t,fr::ForcingModelAndRegion,phys_params) = apply_forcing!(dx,x,t,[fr],phys_params)

#=
The following define how forcing of each type get applied. Each one
calls the forcing model to determine the strength.
=#

function _apply_forcing!(dx,x,t,fcache::ForcingModelAndRegion{<:AreaRegionCache},phys_params)
    @unpack region_cache, fcn = fcache
    @unpack str = region_cache

    fill!(str,0.0)
    fcn(str,x,t,region_cache,phys_params)

    dx .+= str*mask(region_cache)
    return dx
end

function _apply_forcing!(dx,x,t,fcache::ForcingModelAndRegion{<:LineRegionCache},phys_params)
    @unpack region_cache, fcn = fcache
    @unpack str, cache = region_cache

    fill!(str,0.0)
    fcn(str,x,t,region_cache,phys_params)

    fill!(cache.gdata_cache,0.0)
    regularize!(cache.gdata_cache,str,cache)
    dx .+= cache.gdata_cache
    return dx
end

function _apply_forcing!(dx,x,t,fcache::ForcingModelAndRegion{<:PointRegionCache,T},phys_params) where T
    @unpack region_cache, fcn = fcache
    @unpack str, cache = region_cache
    @unpack regop = cache

    fill!(str,0.0)
    fcn(str,x,t,region_cache,phys_params)

    fill!(cache.gdata_cache,0.0)
    regop(cache.gdata_cache,str)
    dx .+= cache.gdata_cache
    return dx
end

function _apply_forcing!(dx,x,t,fcache::ForcingModelAndRegion{<:PointRegionCache,<:Function},phys_params)
    @unpack region_cache, shape, fcn, kwargs = fcache
    @unpack cache = region_cache

    # `shape` is a function in this case, used to obtain the point coordinates
    # Use is to to generate an instantaneous PointRegionCache
    _pts = shape(x,t,region_cache,phys_params)
    typeof(_pts) <: Tuple ? pts = VectorData(_pts...) : pts = _pts
    new_region_cache = PointRegionCache(pts,cache;kwargs...)
    @unpack str, cache = new_region_cache
    @unpack regop = cache

    fill!(str,0.0)
    fcn(str,x,t,region_cache,phys_params)

    fill!(cache.gdata_cache,0.0)
    regop(cache.gdata_cache,str)
    dx .+= cache.gdata_cache
    return dx
end
