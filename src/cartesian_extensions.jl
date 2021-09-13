# These routines should be added to CartesianGrids

import CartesianGrids: cross!

"""
    cross!(C::VectorData,A::ScalarData/VectorData,B::VectorData/ScalarData) -> VectorData

Compute the cross product between the point data `A` and `B`, one of which
is scalar data and treated as an out-of-plane component of a vector, while
the other is in-plane vector data, and return the result as vector data `C`.
"""
function cross!(C::VectorData{N},A::ScalarData{N},B::VectorData{N}) where {N}
    @. C.u = -A*B.v
    @. C.v = A*B.u
    return C
end

function cross!(C::VectorData{N},A::VectorData{N},B::ScalarData{N}) where {N}
    @. C.u = A.v*B
    @. C.v = -A.u*B
    return C
end
