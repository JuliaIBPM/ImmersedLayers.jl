
Δx = 0.02
xlim = (-5.98,5.98)
ylim = (-5.98,5.98)
g = PhysicalGrid(xlim,ylim,Δx)

w = Nodes(Dual,size(g))
dq = Edges(Dual,w)
oc = similar(w)
oc .= 1

n = 150
radius = 1.0
body = Circle(radius,n)
X = VectorData(collect(body))
f = VectorData(X)
ϕ = ScalarData(f)
fill!(ϕ,1.0)

regop = Regularize(X,cellsize(g),I0=origin(g),weights=dlengthmid(body),ddftype=CartesianGrids.Yang3)
Hv = RegularizationMatrix(regop,f,dq)
Hs = RegularizationMatrix(regop,ϕ,w)

@testset "Double Layer" begin

  dlayer = DoubleLayer(body,Hv,weight=1/cellsize(g))
  ϕ .= rand(length(ϕ))

  @test abs(dot(oc,dlayer(ϕ),g)) < 100*eps(1.0)

  dlayer2 = DoubleLayer(body,g,w)

  @test dlayer2(ϕ) == dlayer(ϕ)

end

@testset "Single Layer" begin

  slayer = SingleLayer(body,Hs) #,weight=cellsize(g)^2)
  ϕ .= 1.0

  @test abs(dot(oc,slayer(ϕ),g)-2π*radius) < 2e-3

  slayer2 = SingleLayer(body,g,w)

  @test slayer2(ϕ) == slayer(ϕ)

end

@testset "Masks" begin

  inner = Mask(body,g,w)

  fill!(w,1)

  @test abs(dot(oc,inner(w),g) - π*radius^2) < 1e-3

  outer = ComplementaryMask(inner)

end
