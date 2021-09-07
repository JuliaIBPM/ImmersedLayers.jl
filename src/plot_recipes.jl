using RecipesBase
using ColorTypes


@recipe function f(w::T,cache::BasicILMCache; layers=true) where {T<:GridData}

  framestyle --> :frame
  xlims --> (-Inf,Inf)
  ylims --> (-Inf,Inf)
  aspect_ratio := 1
  legend --> :none
  grid --> false

  @series begin
    trim := 2
    w, cache.g
  end

  if layers
    pts = points(cache)
    @series begin
      pts, cache
    end
  end

end


@recipe function f(w1::T1,w2::T2,cache::BasicILMCache; layers=true) where {T1<:GridData,T2<:GridData}
    @series begin
      w1, cache.g
    end

    @series begin
      linestyle --> :dash
      w2, cache.g
    end

    if layers
      pts = points(cache)
      @series begin
        pts, cache
      end
    end

end

@recipe function f(pts::VectorData,cache::BasicILMCache)
  #x = [pts.u; pts.u[1]]
  #y = [pts.v; pts.v[1]]
  x = pts.u
  y = pts.v
  seriestype --> :scatter
  markersize --> 1
  markershape --> :circle
  markercolor --> :black
  xlims --> (-Inf,Inf)
  ylims --> (-Inf,Inf)
  aspect_ratio := 1
  legend --> :none
  grid --> false
  #linecolor --> :black
  #fillrange --> 0
  #fillalpha --> 0
  #fillcolor --> :black
  x := x
  y := y
  ()
end

@recipe function f(w::T,sys::ILMSystem) where T<:Union{GridData,VectorData}
    w, sys.base_cache
end

@recipe function f(w1::T1,w2::T2,sys::ILMSystem) where {T1<:GridData,T2<:GridData}
    w1, w2, sys.base_cache
end
