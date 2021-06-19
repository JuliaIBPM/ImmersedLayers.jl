using LinearAlgebra

Δx = 0.04
Lx = 4.0
xlim = (-Lx/2,Lx/2)
ylim = (-Lx/2,Lx/2)
g = PhysicalGrid(xlim,ylim,Δx)

RadC = Lx/4
Δs = 1.4*cellsize(g)
body = Circle(RadC,Δs)
X = VectorData(collect(body))

w = Nodes(Dual,size(g))
ϕ = Nodes(Primal,size(g))
q = Edges(Primal,size(g))
f = ScalarData(X)

angs(n) = range(0,2π,length=n+1)[1:n]

@testset "Scalar surface ops" begin
  scache = SurfaceScalarCache(body,g,scaling=GridScaling)

  f .= sin.(angs(length(body)) .- π/4)
  regularize_normal!(q,f,scache)
  normal_interpolate!(f,q,scache)
  @test maximum(f) ≈ 11 atol = 1e-0
  @test minimum(f) ≈ -11 atol = 1e-0

  A = CLinvCT(scache,scale=cellsize(g))

  @test maximum(eigvals(A)) ≈ 0.2 atol = 1e-1
  @test minimum(eigvals(A)) > 0

  A = GLinvD(scache,scale=cellsize(g))
  @test maximum(eigvals(A)) ≈ 0.45 atol = 1e-1

  A = nRTRn(scache)
  @test maximum(svdvals(A)) ≈ 11 atol = 1e-0

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

  A = GLinvD(vcache,scale=cellsize(g))
  @test maximum(eigvals(A)) ≈ 1.7 atol = 1e-1


end
