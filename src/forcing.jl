# Forcing classes (areas, lines, points)

#=
Notes:
- The principle here is that the problem specification will supply the shape
 of the forcing region (area, line) and a model function to apply this forcing.
 These could both be supplied in the forcing keyword, and parsed in the prob_cache
 to generate the forcing cache.
- The forcing model function should be able to make use of other system data, like physical
  parameters
=#
abstract type AbstractForcingModel end
abstract type AbstractRegionCache end



"""
    AreaForcingModel(shape::Union{Body,BodyList},model_function)

Bundles a `shape` (i.e., a `Body` or `BodyList`) and a model_function (a function
that returns the contribution of the forcing to the right-hand side of the PDE)
for area-type forcing
"""
struct AreaForcingModel{RT<:Union{Body,BodyList},MDT} <: AbstractForcingModel
    shape :: RT
    fcn :: MDT
end

"""
    LineForcingModel(shape::Union{Body,BodyList},model_function)

Bundles a `shape` (i.e., a `Body` or `BodyList`) and a `model_function` (a function
that returns the contribution of the forcing to the right-hand side of the PDE)
for line-type forcing
"""
struct LineForcingModel{RT<:Union{Body,BodyList},MDT} <: AbstractForcingModel
    shape :: RT
    fcn :: MDT
end



struct AreaRegionCache{MT,ST,CT} <: AbstractRegionCache
    mask :: MT
    str :: ST
    cache :: CT
end

struct LineRegionCache{ACT,ST,CT} <: AbstractRegionCache
    arccoord :: ACT
    str :: ST
    cache :: CT
end


for f in [:Scalar,:Vector]
    cname = Symbol("Surface"*string(f)*"Cache")
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

end

for f in [:AreaRegionCache,:LineRegionCache]
    @eval $f(shape::Union{Body,BodyList},cache::BasicILMCache{N,SCA};scaling=SCA,kwargs...) where {N,SCA} = $f(cache.g,shape,similar_grid(cache);scaling=scaling,kwargs...)
    @eval $f(g,shape::Body,a...;kwargs...) = $f(g,BodyList([shape]),a...;kwargs...)
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
    ForcingModelAndRegion

A type that holds the forcing model function, region, and cache
# Constructor

`ForcingModelAndRegion(model::AbstractForcingModel,cache::BasicILMCache)`

This form gets called generally when building the extra cache.
""" ForcingModelAndRegion(::AbstractForcingModel,::BasicILMCache)

struct ForcingModelAndRegion{RT<:AbstractRegionCache,MT}
    region_cache :: RT
    fcn :: MT
end

for f in [:Area,:Line]
  modtype = Symbol(string(f)*"ForcingModel")
  regcache = Symbol(string(f)*"RegionCache")
  @eval function ForcingModelAndRegion(model::$modtype,cache::BasicILMCache)
      region_cache = $regcache(model.shape,cache)
      ForcingModelAndRegion(region_cache,model.fcn)
  end
end

"""
    apply_forcing!(dx,x,t,fv::Vector{ForcingModelAndRegion},phys_params)

Return the total contribution of forcing in the vector `fv` to the right-hand side `dx`
based on the current state `x`, time `t`, and physical parameters in `phys_params`.
"""
function apply_forcing!(dx,x,t,fr::Vector{<:ForcingModelAndRegion},phys_params)
    fill!(dx,0.0)
    for f in fr
        _apply_forcing!(dx,x,t,f,phys_params)
    end
    return dx
end

"""
    apply_forcing!(dx,x,t,f::ForcingModelAndRegion,phys_params)

Return the contribution of forcing in `f` to the right-hand side `dx`
based on the current state `x`, time `t`, and physical parameters in `phys_params`.
"""
apply_forcing!(dx,x,t,fr::ForcingModelAndRegion,phys_params) = apply_forcing!(dx,x,t,[fr],phys_params)

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
