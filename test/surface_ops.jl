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

coeff_factor = 1.0
with_inverse = true
L = _get_laplacian(coeff_factor,g,with_inverse,scaling=GridScaling)

angs(n) = range(0,2π,length=n+1)[1:n]

_size(::CartesianGrids.Laplacian{NX,NY}) where {NX,NY} = NX, NY

@testset "Forming a cache" begin
  scache1 = SurfaceScalarCache(body,g,scaling=GridScaling)
  scache2 = SurfaceScalarCache(X,g,scaling=GridScaling)
  scache3 = SurfaceScalarCache(body,g,scaling=GridScaling,L)
  @test normals(scache1) ≈ normals(scache2)
  @test normals(scache1) ≈ normals(scache3)
  @test areas(scache1) ≈ areas(scache2)
  @test areas(scache1) ≈ areas(scache3)
  @test points(scache1) ≈ points(scache2)
  @test points(scache1) ≈ points(scache3)

  @test size(g) == _size(scache1.L)
end

@testset "Fields" begin
  scache1 = SurfaceScalarCache(g,scaling=GridScaling)
  myfun(x,y) = exp(-x)*exp(-y)
  field = SpatialField(myfun)
  evaluate_field!(w,field,scache1)
  xc, yc = coordinates(w,g)

  @test w[104,24] ≈ myfun(xc[24],yc[104])

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

  vd = zeros_gridgradcurl(scache)
  @test typeof(vd) <: Edges{Dual}

  f2 = zeros_griddiv(scache)
  @test typeof(f2) <: Nothing

  f2 = similar_griddiv(scache)
  @test typeof(f2) <: Nothing

  f2 = ones_griddiv(scache)
  @test typeof(f2) <: Nothing

  f2 = x_griddiv(scache)
  @test typeof(f2) <: Nothing

  f2 = y_griddiv(scache)
  @test typeof(f2) <: Nothing

  @test isnothing(scache.Rcurl)
  @test isnothing(scache.Rdiv)

end

@testset "Complex cache" begin
  scache = SurfaceScalarCache(body,g,scaling=GridScaling,dtype=ComplexF64)

  @test eltype(similar_grid(scache)) == ComplexF64
  @test eltype(similar_gridcurl(scache)) == ComplexF64
  @test eltype(similar_gridgrad(scache)) == ComplexF64
  @test eltype(similar_gridgradcurl(scache)) == ComplexF64
  @test eltype(similar_surface(scache)) == ComplexF64


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

  f2 = zeros_griddiv(vcache)
  @test typeof(f2) <: Nodes{Primal}

  f2 = similar_griddiv(vcache)
  @test typeof(f2) <: Nodes{Primal}

  f2 = ones_griddiv(vcache)
  @test typeof(f2) <: Nodes{Primal}

  f2 = x_griddiv(vcache)
  @test typeof(f2) <: Nodes{Primal}

  f2 = y_griddiv(vcache)
  @test typeof(f2) <: Nodes{Primal}

  @test typeof(vcache.Rcurl) <: RegularizationMatrix
  @test typeof(vcache.Rdiv) <: RegularizationMatrix

  v2 = zeros_grid(vcache)
  f2 = ones_surfacescalar(vcache)

  regularize_normal_cross!(v2,f2,vcache)

  vs2 = ones_surface(vcache,2)
  w2 = zeros_gridcurl(vcache)

  regularize_normal_cross!(w2,vs2,vcache)

end



@testset "Masks" begin

  scache = SurfaceScalarCache(body,g,scaling=GridScaling)

  inner = mask(scache)

  oc = similar(ϕ)
  oc .= 1
  @test abs(dot(oc,inner,g) - π*RadC^2) < 2e-3

  outer = complementary_mask(scache)

  pmask = ones_grid(scache)
  vmask = ones_gridgrad(scache)
  wmask = ones_gridcurl(scache)

  mask!(pmask,scache)
  @test abs(integrate(pmask,scache) - π) < 1e-3

  mask!(wmask,scache)
  @test abs(integrate(wmask,scache) - π) < 1e-3

  mask!(vmask.u,scache)
  @test abs(integrate(vmask.u,scache) - π) < 1e-3

  mask!(vmask.v,scache)
  @test abs(integrate(vmask.v,scache) - π) < 1e-3

  vmask = ones_gridgrad(scache)
  mask!(vmask,scache)
  @test all(abs.(integrate(vmask,scache) .- π) .< 1e-3)

  complementary_mask!(vmask,scache)

  vcache = SurfaceVectorCache(body,g,scaling=GridScaling)

  vmask = ones_grid(vcache)
  pmask = ones_griddiv(vcache)
  wmask = ones_gridcurl(vcache)
  dvmask = ones_gridgrad(vcache)

  mask!(pmask,vcache)
  @test abs(integrate(pmask,vcache) - π) < 1e-3

  mask!(wmask,vcache)
  @test abs(integrate(wmask,vcache) - π) < 1e-3

  mask!(vmask,vcache)
  @test all(abs.(integrate(vmask,vcache) .- π) .< 1e-3)

  mask!(dvmask,vcache)
  @test all(abs.(integrate(dvmask,vcache) .- π) .< 1e-3)

  complementary_mask!(dvmask,vcache)


  mask_shape = Rectangle(0.5,0.25,Δs)
  vmask = ones_grid(vcache)
  pmask = ones_griddiv(vcache)
  wmask = ones_gridcurl(vcache)
  dvmask = ones_gridgrad(vcache)

  mask!(pmask,mask_shape,g)
  @test abs(integrate(pmask,g) - 0.5) < 1e-3

  mask!(wmask,mask_shape,g)
  @test abs(integrate(wmask,g) - 0.5) < 1e-3

  mask!(vmask,mask_shape,g)
  @test all(abs.(integrate(vmask,g) .- 0.5) .< 1e-3)

  mask!(dvmask,mask_shape,g)
  @test all(abs.(integrate(dvmask,g) .- 0.5) .< 1e-3)

  complementary_mask!(dvmask,mask_shape,g)

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

  w = zeros_gridcurl(vcache)
  v = zeros_grid(vcache)
  v .= randn(size(v))
  curl!(w,v)
  vdw = zeros_gridcurl(vcache)

  cd_cache = ConvectiveDerivativeCache(zeros_gridgradcurl(vcache))
  convective_derivative!(vdw,v,w,vcache,cd_cache)

  cdr_cache = RotConvectiveDerivativeCache(zeros_gridcurl(vcache))
  vw = zeros_grid(vcache)
  w_cross_v!(vw,w,v,vcache,cdr_cache)

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

@testset "Helmholtz decomposition" begin

  vcache = SurfaceVectorCache(body,g,scaling=GridScaling)
  veccache = VectorFieldCache(vcache)

  w = zeros_gridcurl(vcache)
  ψ = zeros_gridcurl(vcache)
  d = zeros_griddiv(vcache)
  ϕ = zeros_griddiv(vcache)
  v = zeros_grid(vcache)
  dv = zeros_surface(vcache)

  w .= randn(size(w))
  d .= randn(size(d))

  dv .= randn(size(dv))

  vecfield_helmholtz!(v,w,d,dv,(0.0,0.0),vcache,veccache)

  # Test that curl of the resulting vector field is zero
  w2 = zeros_gridcurl(vcache)
  curl!(w2,v,vcache)
  masked_w = zeros_gridcurl(vcache)
  masked_curlv_from_curlv_masked!(masked_w,w2,dv,vcache,veccache.wcache)
  @test maximum(abs.(masked_w[2:end-1,2:end-1] - w[2:end-1,2:end-1])) < 1e-8

  # Test that divergence of the resulting vector field is zero
  d2 = zeros_griddiv(vcache)
  divergence!(d2,v,vcache)
  masked_d = zeros_griddiv(vcache)
  masked_divv_from_divv_masked!(masked_d,d2,dv,vcache,veccache.dcache)
  @test maximum(abs.(masked_d[2:end-1,2:end-1]-d[2:end-1,2:end-1])) < 1e-8

  vectorpotential_from_masked_curlv!(ψ,w,dv,vcache,veccache.wcache)
  scalarpotential_from_masked_divv!(ϕ,d,dv,vcache,veccache.dcache)

  # Tests that no immersed points lead to no problems
  vcache = SurfaceVectorCache(g,scaling=GridScaling)
  veccache = VectorFieldCache(vcache)

  w = zeros_gridcurl(vcache)
  ψ = zeros_gridcurl(vcache)
  d = zeros_griddiv(vcache)
  ϕ = zeros_griddiv(vcache)
  v = zeros_grid(vcache)
  dv = zeros_surface(vcache)

  vectorpotential_from_masked_curlv!(ψ,w,dv,vcache,veccache.wcache)
  scalarpotential_from_masked_divv!(ϕ,d,dv,vcache,veccache.dcache)
  vecfield_helmholtz!(v,w,d,dv,(0.0,0.0),vcache,veccache)
  @test maximum(abs.(ψ)) == 0.0
  @test maximum(abs.(ϕ)) == 0.0
  @test maximum(abs.(v)) == 0.0


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

  prob = BasicScalarILMProblem(g,body,scaling=GridScaling,dtype=ComplexF64)
  sys = construct_system(prob)
  @test eltype(sys.base_cache.gdata_cache) == ComplexF64

  prob = BasicVectorILMProblem(g,body,scaling=GridScaling,dtype=ComplexF64)
  sys = construct_system(prob)
  @test eltype(sys.base_cache.gdata_cache) == ComplexF64

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
   str = [1.0,-1.0]

   function model4!(σ,T,t,fr::PointRegionCache,phys_params)
     σ .= str
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


   function point_function(T,t,fr::PointRegionCache,phys_params)
     return VectorData([-1.2,0.5],[0.5,0.5])
   end
   pfm2 = PointForcingModel(point_function,model4!;ddftype=CartesianGrids.M4prime);

   fcache = ForcingModelAndRegion(pfm,scache)
   fcache2 = ForcingModelAndRegion(pfm2,scache)

   T2 = zeros_grid(scache)
   dT2 = similar_grid(scache)
   t = 0.0

   apply_forcing!(dT,T,t,fcache,phys_params)
   apply_forcing!(dT2,T2,t,fcache2,phys_params)

   @test dT == dT2

   str = [1.0,-1.0,2.0]

   @test_throws DimensionMismatch apply_forcing!(dT2,T2,t,fcache2,phys_params)

   # Area forcing with no shape mask
    afm = AreaForcingModel(model1!)
    fcache = ForcingModelAndRegion(afm,scache)
    apply_forcing!(dT,T,t,fcache,phys_params)

    function model6!(σ,T,t,fr::AreaRegionCache,phys_params)
        σ .= fr.generated_field()
     end
    afm = AreaForcingModel(model6!,spatialfield=SpatialGaussian(0.5,0.1,0,0,10))
    fcache = ForcingModelAndRegion(afm,scache)
    apply_forcing!(dT,T,t,fcache,phys_params)

    @test dT == fcache[1].region_cache.generated_field()

end
