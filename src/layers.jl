# Definitions of layers


"""
    DoubleLayer(b::Body/BodyList,g::PhysicalGrid,u::GridData)

Construct a double-layer operator for a body or bodies `b`. When the
resulting operator acts upon scalar point data `f`, it returns scalar
grid data of the form ``D R_F (f\\circ n)``, where ``D`` is the discrete
divergence operator on grid `g`, ``n`` are the normals associated with
body/bodies `b`, and ``R_F`` is the regularization operator to (edge data on)
`g`. When `f` is of vector point type, then it returns tensor grid data
of the form ``D R_T (f\\times n + n\\times f)``, where ``\\times`` is
the Cartesian product, and ``R_T`` is the regularization operator to
edge gradient data on `g`.
"""
mutable struct DoubleLayer{N,D,G,P} <: LayerType{N}
    weight :: Float64
    nds :: VectorData{N,Float64,D}
    H :: RegularizationMatrix{G,P}
    Pbuf :: P
    Qbuf :: P
    Gbuf :: G
end

function DoubleLayer(body::Union{Body,BodyList},H::RegularizationMatrix{G,P};weight::Float64 = 1.0) where {G,P}
  nrm = normals(body)*weight
  Pbuf = P() #_allocate_point_data(P)
  Qbuf = P()
  Gbuf = G() #_allocate_grid_data(G)
  return DoubleLayer(weight,nrm,H,Pbuf,Qbuf,Gbuf)
end

"""
    DoubleLayer!(dl::DoubleLayer,body::Union{Body,BodyList},g::PhysicalGrid)
"""
function DoubleLayer!(dl::DoubleLayer{N,D,G,P},body::Union{Body,BodyList},H::RegularizationMatrix{G,P}) where {N,D,G,P}
  numpts(body) == N || error("Inconsistent number of points in body")
  dl.nds = normals(body)*dl.weight
  dl.H = H
  return dl
end

#_allocate_point_data(P::Type{VectorData{N}}) where {N} = VectorData(N,dtype=eltype(P))
#_allocate_point_data(P::Type{TensorData{N}}) where {N} = TensorData(N,dtype=eltype(P))
#_allocate_grid_data(G::Type{Edges{C,NX,NY}}) where {C,NX,NY} = Edges(C,(NX,NY),dtype=eltype(G))
#_allocate_grid_data(G::Type{EdgeGradient{C,D,NX,NY}}) where {C,D,NX,NY} = EdgeGradient(C,(NX,NY),dtype=eltype(G))


function DoubleLayer(body::Union{Body,BodyList},g::PhysicalGrid,w::ScalarGridData{NX,NY,T};
                      ddftype=CartesianGrids.Yang3) where {NX,NY,T}
  reg = _get_regularization(body,g,ddftype=ddftype)
  out = RegularizationMatrix(reg,VectorData(numpts(body),dtype=T),Edges(celltype(w),w,dtype=T))
  #return DoubleLayer(body,out, weight = 1/cellsize(g))
  return DoubleLayer(body,out, weight = 1.0)
end

function DoubleLayer(body::Union{Body,BodyList},g::PhysicalGrid,w::VectorGridData{NX,NY,T};
                      ddftype=CartesianGrids.Yang3) where {NX,NY,T}
  reg = _get_regularization(body,g,ddftype=ddftype)
  out = RegularizationMatrix(reg,TensorData(numpts(body),dtype=T),EdgeGradient(celltype(w),w,dtype=T))
  #return DoubleLayer(body,out, weight = 1/cellsize(g))
  return DoubleLayer(body,out, weight = 1.0)
end

function DoubleLayer!(dl::DoubleLayer{N,D,G,P},body::Union{Body,BodyList},g::PhysicalGrid;
                      ddftype=CartesianGrids.Yang3) where {N,D,G,P}

  reg = _get_regularization(body,g,ddftype=ddftype)
  DoubleLayer!(dl,body,RegularizationMatrix(reg,dl.Pbuf,dl.Gbuf))
  return dl
end

# in-place
function (μ::DoubleLayer{N,D,<:Edges{C}})(u::Nodes{C},p::ScalarData{N}) where {N,D,C}
    product!(μ.Pbuf,p,μ.nds)
    divergence!(u,mul!(μ.Gbuf,μ.H,μ.Pbuf))
    return u
end

function (μ::DoubleLayer{N,D,<:EdgeGradient{C}})(u::Edges{C},p::VectorData{N}) where {N,D,C}
    tensorproduct!(μ.Pbuf,p,μ.nds)
    transpose!(μ.Qbuf,μ.Pbuf)
    μ.Pbuf .+= μ.Qbuf
    divergence!(u,mul!(μ.Gbuf,μ.H,μ.Pbuf))
    return u
end


#(μ::DoubleLayer{N})(p::ScalarData{N}) where {N} = divergence(μ.H*(p∘μ.nds))
# out of place
(μ::DoubleLayer{N,D,G,P})(p::ScalarData{N,DG}) where {N,D,G<:Edges{C,NX,NY},P<:PointData{N,DG1},DG} where {C,NX,NY,DG1} =
          μ(Nodes(C,(NX,NY),dtype=DG1),p)

#(μ::DoubleLayer{N})(p::VectorData{N}) where {N} = divergence(μ.H*(p*μ.nds+μ.nds*p))
(μ::DoubleLayer{N,D,G,P})(p::VectorData{N,DG}) where {N,D,G<:EdgeGradient{C,F,NX,NY},P<:PointData{N,DG1},DG} where {C,F,NX,NY,DG1} =
          μ(Edges(C,(NX,NY),dtype=DG1),p)


function (μ::DoubleLayer{N,D,G,P})(p::Number) where {N,D,G<:GridData,P<:PointData}
  ϕ = ScalarData(N,dtype=eltype(P))
  ϕ .= p
  return μ(ϕ)
end

function Base.show(io::IO, H::DoubleLayer{N,D,G,P}) where {N,D,G<:GridData{NX,NY,DG},P<:PointData{N,DP}} where {NX,NY,DG,DP}
    println(io, "Double-layer operator")
    println(io, "  from $N point data of $P type")
    println(io, "  to a $NX x $NY grid of $G data")
end

"""
    SingleLayer(b::Body/BodyList,g::PhysicalGrid,u::GridData)

Construct a single-layer operator for a body or bodies `b`. When the
resulting operator acts upon scalar point data `f`, it returns
scalar grid data of form ``R_C f``, where ``R_C`` is the
regularization operator to (node data on) `g`. When it acts upon
vector point data `f`, it returns vector (edge) grid data of the
form ``R_F f``, where ``R_F`` is the regularization operato to edge
data on `g`.
"""
mutable struct SingleLayer{N,D,G,P} <: LayerType{N}
    weight :: Float64
    ds :: ScalarData{N,Float64,D}
    H :: RegularizationMatrix{G,P}
    Pbuf :: P
    Gbuf :: G
end

function SingleLayer(body::Union{Body,BodyList},H::RegularizationMatrix{G,P};weight::Float64 = 1.0) where {G,P}
  ds = ScalarData(numpts(body))
  ds .= weight
  Pbuf = P()
  Gbuf = G()
  return SingleLayer(weight,ds,H,Pbuf,Gbuf)
end

function SingleLayer!(sl::SingleLayer{N,D,G,P},body::Union{Body,BodyList},H::RegularizationMatrix{G,P}) where {N,D,G,P}
  numpts(body) == N || error("Inconsistent number of points in body")
  sl.H = H
  return sl
end

function SingleLayer(body::Union{Body,BodyList},g::PhysicalGrid,w::GridData{NX,NY,T};
                      ddftype=CartesianGrids.Yang3) where {NX,NY,T}
  reg = _get_regularization(body,g,ddftype=ddftype)
  out = RegularizationMatrix(reg,ScalarData(numpts(body),dtype=T),Nodes(celltype(w),w,dtype=T))
  return SingleLayer(body,out) #,weight=cellsize(g)^2)
end

function SingleLayer!(sl::SingleLayer{N,D,G,P},body::Union{Body,BodyList},g::PhysicalGrid;
                      ddftype=CartesianGrids.Yang3) where {N,D,G,P}

  reg = _get_regularization(body,g,ddftype=ddftype)
  SingleLayer!(sl,body,RegularizationMatrix(reg,sl.Pbuf,sl.Gbuf))
  return sl
end

# In place
function (μ::SingleLayer{N,D,G})(u::G,p::ScalarData{N}) where {N,D,G}
    product!(μ.Pbuf,p,μ.ds)
    mul!(u,μ.H,μ.Pbuf)
    return u
end

# Out of place
(μ::SingleLayer{N,D,G})(p::ScalarData{N}) where {N,D,G} = μ(G(),p)
#(μ::SingleLayer{N})(p::ScalarData{N}) where {N} = μ.H*(p∘μ.ds)


function (μ::SingleLayer{N,D,G,P})(p::Number) where {N,D,G<:GridData,P<:PointData}
  μ.Pbuf .= p
  return μ(μ.Pbuf)
end

function Base.show(io::IO, H::SingleLayer{N,D,G,P}) where {N,D,G<:GridData{NX,NY,DG},P<:PointData{N,DP}} where {NX,NY,DG,DP}
    println(io, "Single-layer operator")
    println(io, "  from $N point data of $P type")
    println(io, "  to a $NX x $NY grid of $G data")
end

abstract type MaskType end

struct Mask{N,NX,NY,G} <: MaskType
  data :: Nodes{G,NX,NY}
end

struct ComplementaryMask{N,NX,NY,G} <: MaskType
  mask :: Mask{N,NX,NY,G}
end

"""
    Mask(b::Body,g::PhysicalGrid,w::GridData)

Returns a `Nodes` mask that sets every grid point equal to 1 if it lies inside the body
`b` and 0 outside. Returns grid data of the same type as `w`, corresponding to
physical grid `g`.
"""
function Mask(dlayer::DoubleLayer{N,D,G,P},g::PhysicalGrid) where {N,D,G<:GridData{NX,NY,DG},P<:PointData{N,DP}} where {NX,NY,DG,DP}
  L = plan_laplacian(CartesianGrids.node_inds(celltype(G),size(g)),with_inverse=true,dtype=DG)
  #return Mask{N,NX,NY,G}(-cellsize(g)^2*(L\dlayer(1)))
  return Mask{N,NX,NY,celltype(G)}(-cellsize(g)*(L\dlayer(1)))
end

Mask(body::Union{Body,BodyList},g::PhysicalGrid,w::GridData{NX,NY,T}) where {NX,NY,T} =
    Mask(DoubleLayer(body,g,w),g)

(m::Mask{N,NX,NY,G})(w::Nodes{G,NX,NY,T}) where {N,NX,NY,G,T} = m.data ∘ w

(m::ComplementaryMask{N,NX,NY,G})(w::Nodes{G,NX,NY,T}) where {N,NX,NY,G,T} = w - m.mask(w)


# Standardize the regularization

_get_regularization(body::Union{Body,BodyList},g::PhysicalGrid;ddftype=CartesianGrids.Yang3) =
     Regularize(VectorData(collect(body)),cellsize(g),I0=origin(g),
                weights=dlengthmid(body),ddftype=ddftype)
