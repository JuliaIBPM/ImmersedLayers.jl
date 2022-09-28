using RecipesBase
using ColorTypes


@recipe function f(w::T,cache::BasicILMCache; layers=true,bodyfill=false) where {T<:GridData}

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
    if typeof(w) <: ScalarGridData
      @series begin
        fill --> bodyfill
        cache
      end
    else
      if typeof(w) <: VectorGridData
        nsub = 2
      elseif typeof(w) <: TensorGridData
        nsub = 4
      end
      for i in 1:nsub
        @series begin
          subplot := i
          fill --> bodyfill
          cache
        end
      end
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
      #pts = points(cache)
      @series begin
        cache
      end
    end

end

@recipe function f(w::T,cache::BasicILMCache; bodyid=1) where {T<:ScalarData}

  legend --> :none
  grid --> false
  @series begin
    view(arcs(cache),cache,bodyid), view(w,cache,bodyid)
  end
end

@recipe function f(w::T,cache::BasicILMCache; bodyid=1) where {T<:VectorData}

  @series begin
    w.u, cache
  end

  @series begin
    w.v, cache
  end
end

@recipe function f(cache::BasicILMCache)
  pts = points(cache)
  #x = [pts.u; pts.u[1]]
  #y = [pts.v; pts.v[1]]
  x = [pts.u;]
  y = [pts.v;]
  #seriestype := :scatter
  markersize --> 1
  markershape --> :circle
  markercolor --> :black
  xlims --> (-Inf,Inf)
  ylims --> (-Inf,Inf)
  aspect_ratio := 1
  legend --> :none
  grid --> false
  linewidth --> 0
  #linecolor --> :black
  #fillrange --> 0
  #fillalpha --> 0
  #fillcolor --> :black
  x := x
  y := y
  ()
end

@recipe function f(sys::ILMSystem)
    sys.base_cache
end

@recipe function f(w::T,sys::ILMSystem) where T<:Union{GridData,ScalarData,VectorData}
    w, sys.base_cache
end

@recipe function f(w1::T1,w2::T2,sys::ILMSystem) where {T1<:GridData,T2<:GridData}
    w1, w2, sys.base_cache
end
