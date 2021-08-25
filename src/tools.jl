import LinearAlgebra: dot, norm
import Base: ones
export normals, areas

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
norm(u::GridData,g::PhysicalGrid) = dot(u,u,g)

"""
    dot(u1::PointData,u2::PointData,ds::ScalarData)

Return the inner product between `u1` and `u2`, weighted by `ds`.
"""
dot(u1::ScalarData{N},u2::ScalarData{N},ds::ScalarData{N}) where {N} = dot(u1,ds∘u2)

dot(u1::VectorData{N},u2::VectorData{N},ds::ScalarData{N}) where {N} =
    dot(u1.u,ds∘u2.u) + dot(u1.v,ds∘u2.v)

"""
    norm(u::PointData,ds::ScalarData)

Return the norm of `u`, weighted by `ds`.
"""
norm(u::PointData{N},ds::ScalarData{N}) where {N} = dot(u,ds,u)

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
