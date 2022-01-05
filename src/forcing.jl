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

struct ForcingModel{MDT,RT}
    model :: MDT
    region :: RT
end

abstract type AbstractRegion end

struct AreaRegion{MT,CT} <: AbstractRegion
    mask :: MT
    cache :: CT
end

struct LineRegion{ACT,CT} <: AbstractRegion
    arccoord :: ACT
    cache :: CT
end

for f in [:Scalar,:Vector]
    cname = Symbol("Surface"*string(f)*"Cache")
    gdtype = Symbol(string(f)*"GridData")
    @eval function AreaRegion(g::PhysicalGrid,shape::BodyList,data_prototype::$gdtype;kwargs...)
        cache = $cname(shape,g;kwargs...)
        m = mask(cache)
        return AreaRegion{typeof(m),typeof(cache)}(m,cache)
    end

    @eval function LineRegion(g::PhysicalGrid,shape::BodyList,data_prototype::$gdtype;kwargs...)
        cache = $cname(shape,g;kwargs...)
        pts = points(shape)

        # This needs to be written specially for closed vs open bodies and for lists
        arccoord = ScalarData(pts)
        arccoord .= accumulate(+,dlengthmid(shape))
        return LineRegion{typeof(arccoord),typeof(cache)}(arccoord,cache)
    end

end

for f in [:AreaRegion,:LineRegion]
    @eval $f(shape::Union{Body,BodyList},cache::BasicILMCache{N,SCA};scaling=SCA,kwargs...) where {N,SCA} = $f(cache.g,shape,similar_grid(cache);scaling=scaling,kwargs...)
    @eval $f(g,shape::Body,a...;kwargs...) = $f(g,BodyList([shape]),a...;kwargs...)
end
