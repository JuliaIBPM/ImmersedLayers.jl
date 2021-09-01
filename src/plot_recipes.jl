using RecipesBase
using ColorTypes
using CartesianGrids
using RigidBodyTools


@recipe function f(w::T,cache::BasicILMCache) where {T<:GridData}
  @series begin
    trim := 2
    w, cache.g
  end
end

@recipe function f(w1::T1,w2::T2,cache::BasicILMCache) where {T1<:GridData,T2<:GridData}
    @series begin
      w1, cache.g
    end

    @series begin
      linestyle --> :dash
      w2, cache.g
    end

end
