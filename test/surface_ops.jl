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
  @test minimum(vs.u) ≈ -39 atol = 1e-0
  @test maximum(vs.u) ≈ 39 atol = 1e-0

  surface_divergence!(q,vs,vcache)
  surface_grad!(vs,q,vcache)

  A = create_GLinvD(vcache,scale=cellsize(g))
  @test maximum(abs.(eigvals(A))) ≈ 1.7 atol = 1e-1


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
  scache = SurfaceScalarCache(body,g,scaling=GridScaling)

  q .= randn(size(q))
  cv_cache = ConvectiveDerivativeCache(EdgeGradient(Primal,q))
  qdq = similar(q)
  qdq .= 0.0
  convective_derivative!(qdq,q,scache,cv_cache)

  qdq2 = convective_derivative(q,scache)

  @test qdq == qdq2
end

@testset "Problem specification" begin

  prob = BasicScalarILMProblem(g,scaling=GridScaling)
  @test typeof(prob.bodies) <: BodyList

  prob = BasicScalarILMProblem(g,body,scaling=GridScaling)
  @test typeof(prob.bodies) <: BodyList

  sys = ImmersedLayers.__init(prob)

  scache = SurfaceScalarCache(body,g,scaling=GridScaling)

  @test typeof(sys.base_cache) == typeof(scache)

  vprob = BasicVectorILMProblem(g,body,scaling=GridScaling)
  vsys = ImmersedLayers.__init(vprob)

  vcache = SurfaceVectorCache(body,g,scaling=GridScaling)

  @test typeof(vsys.base_cache) == typeof(vcache)

  p = rand()
  prob = BasicScalarILMProblem(g,body,scaling=GridScaling,phys_params=p)
  sys = ImmersedLayers.__init(prob)
  @test sys.phys_params == p

  Re = rand()
  p = Dict("Re"=>Re)
  prob = BasicScalarILMProblem(g,body,scaling=GridScaling,phys_params=p)
  sys = ImmersedLayers.__init(prob)
  @test sys.phys_params["Re"] == Re

  my_bc_func(x) = x
  prob = BasicScalarILMProblem(g,body,scaling=GridScaling,bc=my_bc_func)
  sys = ImmersedLayers.__init(prob)
  @test sys.bc == my_bc_func

  my_f_func(x) = x
  prob = BasicScalarILMProblem(g,body,scaling=GridScaling,forcing=my_f_func)
  sys = ImmersedLayers.__init(prob)
  @test sys.forcing == my_f_func

end

@testset "Problem regeneration" begin

  my_f_func(x) = x
  prob = BasicScalarILMProblem(g,body,scaling=GridScaling,forcing=my_f_func)
  sys = ImmersedLayers.__init(prob)

  prob2 = regenerate_problem(sys,sys.base_cache.bl)
  @test prob.g == prob2.g
  @test prob.bodies == prob2.bodies
  @test prob.phys_params == prob2.phys_params
  @test prob.bc == prob2.bc
  @test prob.forcing == prob2.forcing
  @test prob.timestep_func == prob2.timestep_func


end

@testset "System update" begin
  

end
