import LinearAlgebra: dot, norm
import RigidBodyTools: view
import Base: ones
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

"""
    view(v::VectorData,bl::BodyList,i::Int)

Provide a view of the range of values in `VectorData` `v`, corresponding to the
points of the body with index `i` in a BodyList `bl`.
"""
@inline view(v::VectorData,bl::BodyList,i::Int) = CatView(view(v.u,bl,i),view(v.v,bl,i))


"""
    dot(u1::GridData,u2::GridData,g::PhysicalGrid)

Return the inner product between `u1` and `u2` weighted by the volume (area)
of the cell in grid `g`.
"""
function LinearAlgebra.dot(u1::GridData{NX,NY},u2::GridData{NX,NY},g::PhysicalGrid) where {NX,NY}
    @assert (NX,NY) == size(g)
    dot(u1,u2)*volume(g)
end

"""
    norm(u::GridData,g::PhysicalGrid)

Return the norm of `u`, weighted by the volume (area)
of the cell in grid `g`.
"""
norm(u::GridData,g::PhysicalGrid) = sqrt(dot(u,u,g))

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

"""
    ones(u::ScalarData)

Returns `ScalarData` of the same type as `u` filled with ones.
"""
function ones(u::ScalarData{N}) where {N}
  o = similar(u)
  o .= 1
  o
end

"""
    ones(u::VectorData,dim::Int)

Returns `VectorData` of the same type as `u`, filled with ones in
component `dim`.
"""
function ones(u::VectorData{N},dim::Integer) where {N}
  o = zero(u)
  ocomp = getfield(o,dim+1) # offset
  ocomp .= 1
  o
end

"""
    ones(u::ScalarGridData)

Returns `ScalarGridData` of the same type as `u` filled with ones.
"""
function ones(u::ScalarGridData)
  o = similar(u)
  o .= 1
  o
end

"""
    ones(u::VectorGridData,dim::Int)

Returns `VectorGridData` of the same type as `u`, filled with ones in
component `dim`.
"""
function ones(u::VectorGridData,dim::Integer)
  o = zero(u)
  ocomp = getfield(o,dim+1) # offset
  ocomp .= 1
  o
end
