using LinearAlgebra

Δx = 0.02
xlim = (-5.98,5.98)
ylim = (-5.98,5.98)
g = PhysicalGrid(xlim,ylim,Δx)

w = Nodes(Dual,size(g))
q = Edges(Dual,w)
oc = similar(w)
oc .= 1

radius = 1.0
body = Circle(radius,1.5*Δx)
X = VectorData(collect(body))
ds = areas(body)

f = VectorData(X)
ϕ = ScalarData(f)
ϕ .= 1

os = similar(ϕ)
os .= 1

@testset "Inner Products" begin
   w .= 1
   u = similar(w)
   u .= 1
   @test dot(w,u,g) == norm(w,g)

   u = Nodes(Dual,5,5)
   @test_throws MethodError dot(w,u,g)
   @test_throws AssertionError dot(u,u,g)

   @test norm(w,g) == (xlim[2]-xlim[1])*(ylim[2]-ylim[1])

   @test isapprox(dot(os,ds,os),2π,atol=1e-3)

end

@testset "Regularization and Interpolation" begin
  reg = Regularize(X,Δx,I0=origin(g),weights=dlengthmid(body),ddftype=CartesianGrids.Yang3)


  Rc = RegularizationMatrix(reg,ϕ,oc)
  Ec = InterpolationMatrix(reg,oc,ϕ)

  @test isapprox(abs(dot(oc,Rc*os,g) - dot(os,ds,os)),0,atol=1e-14)

  u = similar(w)
  u .= randn(size(u))
  ϕ .= randn(size(ϕ))
  @test isapprox(dot(u,Rc*ϕ,g),dot(Ec*u,ds,ϕ),atol=1e-14)

  Rf = RegularizationMatrix(reg,f,q);
  Ef = InterpolationMatrix(reg,q,f);

  f.u .= 1;
  q.u .= 1;
  @test isapprox(abs(dot(q.u,(Rf*f).u,g)),sum(ds),atol=1e-14)
  @test isapprox(abs(dot(q.u,(Rf*f).u,g)),2π,atol=1e-3)



end
