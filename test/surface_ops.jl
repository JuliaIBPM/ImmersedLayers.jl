using LinearAlgebra

Δx = 0.04
Lx = 4.0
xlim = (-Lx/2,Lx/2)
ylim = (-Lx/2,Lx/2)
g = PhysicalGrid(xlim,ylim,Δx)

RadC = 1.0
Δs = 1.4*cellsize(g)
body = Circle(RadC,Δs)
X = VectorData(collect(body,endpoints=true))

w = Nodes(Dual,size(g))
ϕ = Nodes(Primal,size(g))
q = Edges(Primal,size(g))
f = ScalarData(X)

angs(n) = range(0,2π,length=n+1)[1:n]

@testset "Forming a cache" begin
  scache1 = SurfaceScalarCache(body,g,scaling=GridScaling)
  scache2 = SurfaceScalarCache(X,g,scaling=GridScaling)
  @test normals(scache1) ≈ normals(scache2)
  @test areas(scache1) ≈ areas(scache2)
  @test points(scache1) ≈ points(scache2)
end

@testset "Scalar surface ops" begin
  scache = SurfaceScalarCache(body,g,scaling=GridScaling)

  gd = similar_grid(scache)
  @test typeof(gd) == typeof(scache.gdata_cache)

  sd = similar_surface(scache)
  @test typeof(sd) == typeof(scache.sdata_cache)

  f .= sin.(angs(length(body)) .- π/4)
  regularize_normal!(q,f,scache)
  normal_interpolate!(f,q,scache)
  @test maximum(f) ≈ 11 atol = 1e-0
  @test minimum(f) ≈ -11 atol = 1e-0

  A = create_CLinvCT(scache,scale=cellsize(g))

  @test maximum(abs.(eigvals(A))) ≈ 0.2 atol = 1e-1

  A = create_GLinvD(scache,scale=cellsize(g))
  @test maximum(abs.(eigvals(A))) ≈ 0.45 atol = 1e-1

  A = create_nRTRn(scache)
  @test maximum(svdvals(A)) ≈ 11 atol = 1e-0

  # Create additional regularization and interpolation ops
  w = ones_gridcurl(scache)
  Rn = RegularizationMatrix(scache,scache.sdata_cache,w)
  En = InterpolationMatrix(scache,w,scache.sdata_cache)

  f .= En*w

end

dq = EdgeGradient(Primal,size(g))
vs = VectorData(X)

@testset "Vector surface ops" begin
  vcache = SurfaceVectorCache(body,g,scaling=GridScaling)

  vs.u .= sin.(angs(length(body)) .- π/4)

  regularize_normal!(dq,vs,vcache)
  normal_interpolate!(vs,dq,vcache)

  fill!(dq,0.0)
  vs.u .= sin.(angs(length(body)) .- π/4)
  regularize_normal_symm!(dq,vs,vcache)
  normal_interpolate_symm!(vs,dq,vcache)
  @test minimum(vs.u) ≈ -22.5 atol = 1e-0
  @test maximum(vs.u) ≈ 22.5 atol = 1e-0

  surface_divergence!(q,vs,vcache)
  surface_grad!(vs,q,vcache)

  surface_divergence_symm!(q,vs,vcache)
  surface_grad_symm!(vs,q,vcache)

  A = create_GLinvD(vcache,scale=cellsize(g))
  @test maximum(abs.(eigvals(A))) ≈ 0.45 atol = 1e-1


end



@testset "Masks" begin

  scache = SurfaceScalarCache(body,g,scaling=GridScaling)

  inner = mask(scache)

  oc = similar(ϕ)
  oc .= 1
  @test abs(dot(oc,inner,g) - π*RadC^2) < 2e-3

  outer = complementary_mask(scache)



end

@testset "Grid operations" begin
  vcache = SurfaceVectorCache(body,g,scaling=GridScaling)

  q .= randn(size(q))
  cv_cache = ConvectiveDerivativeCache(EdgeGradient(Primal,q))
  cv_cache2 = ConvectiveDerivativeCache(vcache)
  qdq = similar(q)
  qdq .= 0.0
  convective_derivative!(qdq,q,vcache,cv_cache2)

  qdq2 = convective_derivative(q,vcache)

  @test qdq == qdq2
end

@testset "Convenience fields" begin
  scache = SurfaceScalarCache(body,g,scaling=GridScaling)

  x = x_grid(scache)
  y = y_grid(scache)
  @test typeof(x) == typeof(y) == typeof(scache.gdata_cache)

  x = x_gridcurl(scache)
  y = y_gridcurl(scache)
  @test typeof(x) == typeof(y) == typeof(scache.gcurl_cache)

  x = x_gridgrad(scache)
  y = y_gridgrad(scache)
  @test typeof(x) == typeof(y) == typeof(scache.gsnorm_cache)

  vcache = SurfaceVectorCache(body,g,scaling=GridScaling)

  x = x_grid(vcache)
  y = y_grid(vcache)
  @test typeof(x) == typeof(y) == typeof(vcache.gdata_cache)

  x = x_gridcurl(vcache)
  y = y_gridcurl(vcache)
  @test typeof(x) == typeof(y) == typeof(vcache.gcurl_cache)

  x = x_gridgrad(vcache)
  y = y_gridgrad(vcache)
  @test typeof(x) == typeof(y) == typeof(vcache.gsnorm_cache)


end

@testset "Problem specification" begin

  prob = BasicScalarILMProblem(g,scaling=GridScaling)
  @test typeof(prob.bodies) <: BodyList

  prob = BasicScalarILMProblem(g,body,scaling=GridScaling)
  @test typeof(prob.bodies) <: BodyList

  sys = construct_system(prob)

  scache = SurfaceScalarCache(body,g,scaling=GridScaling)

  @test typeof(sys.base_cache) == typeof(scache)

  vprob = BasicVectorILMProblem(g,body,scaling=GridScaling)
  vsys = construct_system(vprob)

  vcache = SurfaceVectorCache(body,g,scaling=GridScaling)

  @test typeof(vsys.base_cache) == typeof(vcache)

  p = rand()
  prob = BasicScalarILMProblem(g,body,scaling=GridScaling,phys_params=p)
  sys = construct_system(prob)
  @test sys.phys_params == p

  Re = rand()
  p = Dict("Re"=>Re)
  prob = BasicScalarILMProblem(g,body,scaling=GridScaling,phys_params=p)
  sys = construct_system(prob)
  @test sys.phys_params["Re"] == Re

  my_bc_func(x) = x
  prob = BasicScalarILMProblem(g,body,scaling=GridScaling,bc=my_bc_func)
  sys = construct_system(prob)
  @test sys.bc == my_bc_func

  my_f_func(x) = x
  prob = BasicScalarILMProblem(g,body,scaling=GridScaling,forcing=my_f_func)
  sys = construct_system(prob)
  @test sys.forcing == my_f_func

end

@testset "Problem regeneration" begin

  my_f_func(x) = x
  prob = BasicScalarILMProblem(g,body,scaling=GridScaling,forcing=my_f_func)
  sys = construct_system(prob)

  prob2 = ImmersedLayers.regenerate_problem(sys,sys.base_cache.bl)
  @test prob.g == prob2.g
  @test prob.bodies == prob2.bodies
  @test prob.phys_params == prob2.phys_params
  @test prob.bc == prob2.bc
  @test prob.forcing == prob2.forcing
  @test prob.timestep_func == prob2.timestep_func


end

@testset "System update" begin


end

@testset "Forcing regions" begin
  Δs = 1.4*cellsize(g)

  scache = SurfaceScalarCache(body,g,scaling=GridScaling)

  phys_params = Dict("areaheater1_flux" => 3.0,
                      "lineheater_flux" => -2.0,
                      "areaheater2_temp" => 1.5,
                       "areaheater2_coeff" => 2.0)

  fregion1 = Circle(0.2,Δs)
  T = RigidTransform((0.5,0.0),0.0)
  T(fregion1)
  function model1!(σ,T,t,fr::AreaRegionCache,phys_params)
      σ .= phys_params["areaheater1_flux"]
   end
   afm = AreaForcingModel(fregion1,model1!)

   fregion2 = Square(0.5,Δs)
   T = RigidTransform((0.0,1.0),0.0)
   T(fregion2)
   function model2!(σ,T,t,fr::LineRegionCache,phys_params)
     σ .= phys_params["lineheater_flux"]
    end
   lfm = LineForcingModel(fregion2,model2!)

   fregion3 = Polygon([0.0,1.0,0.5],[0.0,0.0,0.5],Δs)
   T = RigidTransform((-1.0,-1.0),π/4)
   T(fregion3)
   function model3!(σ,T,t,fr::AreaRegionCache,phys_params)
     σ .= phys_params["areaheater2_coeff"]*(phys_params["areaheater2_temp"] - T)
   end
   afm2 = AreaForcingModel(fregion3,model3!)

   pts = VectorData([-1.2,0.5],[0.5,0.5])
   function model4!(σ,T,t,fr::PointRegionCache,phys_params)
     σ[1] = 1.0
     σ[2] = -1.0
     return σ
   end
   pfm = PointForcingModel(pts,model4!;ddftype=CartesianGrids.M4prime);


   fcache = ForcingModelAndRegion([afm,lfm,afm2,pfm],scache)

   @test typeof(fcache[1].region_cache.mask) <: ScalarGridData

   @test typeof(fcache[4].region_cache.cache.regop.ddf) == DDF{CartesianGrids.M4prime, 1.0}

   T = zeros_grid(scache)
   dT = similar_grid(scache)
   t = 0.0

   apply_forcing!(dT,T,t,fcache,phys_params)



end
