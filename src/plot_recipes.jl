using RecipesBase
using ColorTypes
#using CartesianGrids
#using RigidBodyTools


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
    #x = [pts.u; pts.u[1]]
    #y = [pts.v; pts.v[1]]
    x = pts.u
    y = pts.v
    @series begin
      seriestype --> :scatter
      markersize --> 1
      markershape --> :circle
      markercolor --> :black
      #linecolor --> :black
      #fillrange --> 0
      #fillalpha --> 0
      #fillcolor --> :black
      x := x
      y := y
      ()
    end
  end



end

@recipe function f(w::T,sys::ILMSystem) where {T<:GridData}
  @series begin
    trim := 2
    w, sys.base_cache
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

@recipe function f(w1::T1,w2::T2,sys::ILMSystem) where {T1<:GridData,T2<:GridData}
    @series begin
      w1, sys.base_cache
    end

    @series begin
      linestyle --> :dash
      w2, sys.base_cache
    end

end

#=
@recipe function f(b::Body)
    x = [b.x; b.x[1]]
    y = [b.y; b.y[1]]
    linecolor --> mygreen
    fillrange --> 0
    fillcolor --> mygreen
    aspect_ratio := 1
    legend := :none
    grid := false
    x := x
    y := y
    ()
end
=#
