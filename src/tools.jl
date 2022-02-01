using CatViews

import LinearAlgebra: dot, norm
import RigidBodyTools: view
import CartesianGrids: integrate
import Base: ones, copyto!
export points, normals, areas


"""
    points(b::Body/BodyList)

Return `VectorData` filled with the coordinates of the points
associated with `b`.
"""
@inline points(b::Union{Body,BodyList}) = VectorData(collect(b))

"""
    normals(b::Body/BodyList)

Return `VectorData` filled with the normal vectors (via midpoint rule)
associated with `b`.
"""
function normals(b::Union{Body,BodyList})
  nrm = VectorData(numpts(b))
  nx, ny = normalmid(b)
  nrm.u .= nx
  nrm.v .= ny
  nrm
end

"""
    areas(b::Body/BodyList)

Return `ScalarData` filled with the 1-d areas (via midpoint rule)
associated with `b`.
"""
areas(b::Union{Body,BodyList}) = ScalarData(dlengthmid(b))

## Tools for partitioned operations via body lists

"""
    view(v::VectorData,bl::BodyList,i::Int)

Provide a view of the range of values in `VectorData` `v`, corresponding to the
points of the body with index `i` in a BodyList `bl`.
"""
@inline view(v::VectorData,bl::BodyList,i::Int) = CatView(view(v.u,bl,i),view(v.v,bl,i))

"""
    copyto!(u::PointData,v::PointData,bl::BodyList,i::Int)

Copy the data in the elements of `v` associated with body `i` in body list `bl` to
the corresponding elements in `u`. These data must be of the same type (e.g.,
`ScalarData` or `VectorData`) and have the same length.
""" copyto!(u::PointData,v::PointData,bl::BodyList,i::Int)

for f in [:ScalarData,:VectorData]
  @eval function copyto!(u::$f{N},v::$f{N},bl::BodyList,i::Int) where {N}
    ui = view(u,bl,i)
    vi = view(v,bl,i)
    ui .= vi
    return u
  end
end


"""
    copyto!(u::ScalarData,v::AbstractVector,bl::BodyList,i::Int)

Copy the data in `v` to the elements in `u` associated with body `i` in body list `bl`.
`v` must have the same length as this subarray of `u` associated with `i`.
"""
function copyto!(u::ScalarData{N},v::AbstractVector,bl::BodyList,i::Int) where {N}
    ui = view(u,bl,i)
    @assert length(v) == length(ui) "Lengths are incompatible for copyto!"
    ui .= v
    return u
end

### GRID OPERATIONS


## Dot, norm, integrate

"""
    dot(u1::GridData,u2::GridData,g::PhysicalGrid)

Return the inner product between `u1` and `u2` weighted by the volume (area)
of the cell in grid `g`.
"""
function dot(u1::GridData{NX,NY},u2::GridData{NX,NY},g::PhysicalGrid) where {NX,NY}
    @assert (NX,NY) == size(g)
    dot(u1,u2)*volume(g)
end

"""
    norm(u::GridData,g::PhysicalGrid)

Return the L2 norm of `u`, weighted by the volume (area)
of the cell in grid `g`.
"""
norm(u::GridData,g::PhysicalGrid) = sqrt(dot(u,u,g))

"""
    ones(u::GridData)

Returns `GridData` of the same type as `u` filled with ones.
"""
function ones(u::GridData)
  o = similar(u)
  o .= 1
  o
end

"""
    ones(u::VectorGridData/TensorGridData,dim::Int)

Returns grid data of the same type as `u`, filled with ones in
component `dim`.
"""
function ones(u::Union{VectorGridData,TensorGridData},dim::Integer)
  o = zero(u)
  ocomp = getfield(o,dim+1) # offset
  ocomp .= 1
  o
end


### POINT OPERATIONS

## Inner products

"""
    dot(u1::PointData,u2::PointData,ds::ScalarData)

Return the inner product between `u1` and `u2`, weighted by `ds`.
"""
dot(u1::ScalarData{N},u2::ScalarData{N},ds::ScalarData{N}) where {N} = dot(u1,ds∘u2)

dot(u1::VectorData{N},u2::VectorData{N},ds::ScalarData{N}) where {N} =
    dot(u1.u,u2.u,ds) + dot(u1.v,u2.v,ds)


"""
    dot(u1::PointData,u2::PointData,ds::ScalarData,bl::BodyList,i)

Return the inner product between `u1` and `u2`, weighted by `ds`,
for only the data associated with body `i` in body list `bl`.
"""
dot(u1::ScalarData{N},u2::ScalarData{N},ds::ScalarData{N},bl::BodyList,i::Int) where {N} =
    dot(ScalarData(view(u1,bl,i)),ScalarData(view(u2,bl,i)),ScalarData(view(ds,bl,i)))

dot(u1::VectorData{N},u2::VectorData{N},ds::ScalarData{N},bl::BodyList,i::Int) where {N} =
    dot(ScalarData(view(u1.u,bl,i)),ScalarData(view(u2.u,bl,i)),ScalarData(view(ds,bl,i))) +
    dot(ScalarData(view(u1.v,bl,i)),ScalarData(view(u2.v,bl,i)),ScalarData(view(ds,bl,i)))


# Extend the inner products that do not scale with surface areas.
dot(u1::ScalarData{N},u2::ScalarData{N},bl::BodyList,i::Int) where {N} =
    dot(ScalarData(view(u1,bl,i)),ScalarData(view(u2,bl,i)))

dot(u1::VectorData{N},u2::VectorData{N},bl::BodyList,i::Int) where {N} =
    dot(ScalarData(view(u1.u,bl,i)),ScalarData(view(u2.u,bl,i))) +
    dot(ScalarData(view(u1.v,bl,i)),ScalarData(view(u2.v,bl,i)))

## Norms

"""
    norm(u::PointData,ds::ScalarData)

Return the norm of `u`, weighted by `ds`.
"""
norm(u::PointData{N},ds::ScalarData{N}) where {N} = sqrt(dot(u,u,ds))

"""
    norm(u::PointData,ds::ScalarData,bl::BodyList,i)

Return the norm of `u`, weighted by `ds`, for body `i` in body list `bl`
"""
norm(u::PointData{N},ds::ScalarData{N},bl::BodyList,i::Int) where {N} = sqrt(dot(u,u,ds,bl,i))

# Extend the norm that does not scale with surface areas.
norm(u::PointData{N},bl::BodyList,i::Int) where {N} = sqrt(dot(u,u,bl,i))

## Integrals

"""
    integrate(u::PointData,ds::ScalarData)

Calculate the discrete surface integral of data `u`, using the
surface element areas in `ds`. This uses trapezoidal rule quadrature.
If `u` is `VectorData`, then this returns a vector of the integrals in
each coordinate direction.
"""
integrate(u::ScalarData{N},ds::ScalarData{N}) where {N} = dot(u,ds)

integrate(u::VectorData{N},ds::ScalarData{N}) where {N} = [dot(u.u,ds),dot(u.v,ds)]

"""
    integrate(u::PointData,ds::ScalarData,bl::BodyList,i::Int)

Calculate the discrete surface integral of scalar data `u`, restricting the
integral to body `i` in body list `bl`, using the
surface element areas in `ds`. This uses trapezoidal rule quadrature.
If `u` is `VectorData`, then this returns a vector of the integrals in
each coordinate direction.
"""
integrate(u::ScalarData{N},ds::ScalarData{N},bl::BodyList,i::Int) where {N} = dot(u,ds,bl,i)

integrate(u::VectorData{N},ds::ScalarData{N},bl::BodyList,i::Int) where {N} = [dot(u.u,ds,bl,i); dot(u.v,ds,bl,i)]



"""
    ones(u::PointData)

Returns `PointData` of the same type as `u` filled with ones.
"""
function ones(u::PointData{N}) where {N}
  o = similar(u)
  o .= 1
  o
end

"""
    ones(u::VectorData/TensorData,dim::Int)

Returns point data of the same type as `u`, filled with ones in
component `dim`.
"""
function ones(u::Union{VectorData,TensorData},dim::Integer)
  o = zero(u)
  ocomp = getfield(o,dim+1) # offset
  ocomp .= 1
  o
end
