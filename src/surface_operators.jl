"""
    regularize_normal!(q::Edges{Primal},f::ScalarData,cache::SurfaceCache)

The operation ``R_f n\\circ``, which maps scalar surface data `f` (like
a jump in scalar potential) to grid data `q` (like velocity). This is the adjoint
to `normal_interpolate!`.
"""
@inline regularize_normal!(q::Edges{Primal},f::ScalarData,cache::SurfaceCache) = regularize_normal!(q,f,cache.nrm,cache.R,cache.sv_cache)

function regularize_normal!(q::Edges{Primal,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,sv_cache::VectorData{N}) where {NX,NY,N}
    product!(sv_cache,nrm,f)
    q .= Rf*sv_cache
end

"""
    regularize_normal!(qt::EdgeGradient{Primal},v::VectorData,cache::SurfaceCache)

The operation ``R_T n\\circ``, which maps scalar vector data `v` (like
a jump in velocity) to grid data `qt` (like velocity-normal tensor). This is the adjoint
to `normal_interpolate!`.
"""
@inline regularize_normal!(q::EdgeGradient{Primal},f::VectorData,cache::SurfaceCache) = regularize_normal!(q,f,cache.nrm,cache.R,cache.sv_cache,cache.sv2_cache)

function regularize_normal!(q::EdgeGradient{Primal,Dual,NX,NY},f::VectorData{N},nrm::VectorData{N},Rf::RegularizationMatrix,sv_cache::TensorData{N},sv2_cache::TensorData{N}) where {NX,NY,N}
    tensorproduct!(sv_cache,nrm,f)
    transpose!(sv2_cache,sv_cache)
    sv_cache .+= sv2_cache
    q .= Rf*sv_cache
end


"""
    normal_interpolate!(vn::ScalarData,q::Edges{Primal},cache::SurfaceCache)

The operation ``n \\cdot I_f``, which maps grid data `q` (like velocity) to scalar
surface data `vn` (like normal component of surface velocity). This is the
adjoint to `regularize_normal!`.
"""
@inline normal_interpolate!(vn::ScalarData,q::Edges{Primal},cache::SurfaceCache) = normal_interpolate!(vn,q,cache.nrm,cache.E,cache.sv_cache)

function normal_interpolate!(vn::ScalarData{N},q::Edges{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,sv_cache::VectorData{N}) where {NX,NY,N}
    sv_cache .= Ef*q
    pointwise_dot!(vn,nrm,sv_cache)
end

"""
    normal_interpolate!(τ::VectorData,A::EdgeGradient{Primal},cache::SurfaceCache)

The operation ``n \\cdot I_T (A + A^T)``, which maps grid tensor data `A` (like velocity gradient tensor) to vector
surface data `τ` (like traction). This is the adjoint to `regularize_normal!`.
"""
@inline normal_interpolate!(vn::VectorData,q::EdgeGradient{Primal},cache::SurfaceCache) = normal_interpolate!(vn,q,cache.nrm,cache.E,cache.gv2_cache,cache.sv_cache)

function normal_interpolate!(vn::VectorData{N},q::EdgeGradient{Primal,Dual,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,gv_cache::EdgeGradient{Primal,Dual,NX,NY},sv_cache::TensorData{N}) where {NX,NY,N}
    transpose!(gv_cache,q)
    gv_cache .+= q
    sv_cache .= Ef*gv_cache
    pointwise_dot!(vn,nrm,sv_cache)
end

"""
    surface_curl!(w::Nodes{Dual},f::ScalarData,cache::SurfaceCache)

The operation ``C_s^T = C^T R_f n\\circ``, which maps scalar surface data `f` (like
a jump in scalar potential) to grid data `w` (like vorticity). This is the adjoint
to ``C_s``, also given by `surface_curl!` (but with arguments switched).
"""
@inline surface_curl!(w::Nodes{Dual},f::ScalarData,cache::SurfaceCache) = surface_curl!(w,f,cache.nrm,cache.R,cache.gv_cache,cache.sv_cache)

function surface_curl!(w::Nodes{Dual,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::Edges{Primal,NX,NY},sv_cache::VectorData{N}) where {NX,NY,N}
    regularize_normal!(q_cache,f,nrm,Rf,sv_cache)
    curl!(w,q_cache)
end

"""
    surface_curl!(vn::ScalarData,s::Nodes{Dual},cache::SurfaceCache)

The operation ``C_s = n \\cdot I_f C``, which maps grid data `s` (like
streamfunction) to scalar surface data `vn` (like normal component of velocity).
This is the adjoint to ``C_s^T``, also given by `surface_curl!`, but with
arguments switched.
"""
@inline surface_curl!(vn::ScalarData,s::Nodes{Dual},cache::SurfaceCache) = surface_curl!(vn,s,cache.nrm,cache.E,cache.gv_cache,cache.sv_cache)

function surface_curl!(vn::ScalarData{N},s::Nodes{Dual,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,q_cache::Edges{Primal,NX,NY},sv_cache::VectorData{N}) where {NX,NY,N}
    fill!(q_cache,0.0)
    curl!(q_cache,s)
    normal_interpolate!(vn,q_cache,nrm,Ef,sv_cache)
end


"""
    surface_divergence!(Θ::Nodes{Primal},f::ScalarData,cache::SurfaceCache)

The operation ``D_s = D R_f n \\circ``, which maps surface scalar data `f` (like
jump in scalar potential) to grid data `Θ` (like dilatation, i.e. divergence of velocity).
"""
@inline surface_divergence!(θ::Nodes{Primal},f::ScalarData,cache::SurfaceCache) = surface_divergence!(θ,f,cache.nrm,cache.R,cache.gv_cache,cache.sv_cache)

function surface_divergence!(θ::Nodes{Primal,NX,NY},f::ScalarData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::Edges{Primal,NX,NY},sv_cache::VectorData{N}) where {NX,NY,N}
    regularize_normal!(q_cache,f,nrm,Rf,sv_cache)
    divergence!(θ,q_cache)
end

"""
    surface_divergence!(v::Edges{Primal},dv::VectorData,cache::SurfaceCache)

The operation ``D_s = D R_f (n \\circ \\cdot + \\cdot \\circ n)``, which maps surface vector data `v` (like
jump in velocity) to grid data `v` (like velocity).
"""
@inline surface_divergence!(θ::Edges{Primal},f::VectorData,cache::SurfaceCache) = surface_divergence!(θ,f,cache.nrm,cache.R,cache.gv_cache,cache.sv_cache,cache.sv2_cache)

function surface_divergence!(θ::Edges{Primal,NX,NY},f::VectorData{N},nrm::VectorData{N},Rf::RegularizationMatrix,q_cache::EdgeGradient{Primal,Dual,NX,NY},sv_cache::TensorData{N},sv2_cache::TensorData{N}) where {NX,NY,N}
    regularize_normal!(q_cache,f,nrm,Rf,sv_cache,sv2_cache)
    divergence!(θ,q_cache)
end

"""
    surface_grad!(vn::ScalarData,ϕ::Nodes{Primal},cache::SurfaceCache)

The operation ``G_s = n \\cdot I_f G``, which maps grid data `ϕ` (like
scalar potential) to scalar surface data (like normal component of velocity).
"""
@inline surface_grad!(vn::ScalarData,ϕ::Nodes{Primal},cache::SurfaceCache) = surface_grad!(vn,ϕ,cache.nrm,cache.E,cache.gv_cache,cache.sv_cache)

function surface_grad!(vn::ScalarData{N},ϕ::Nodes{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,q_cache::Edges{Primal,NX,NY},sv_cache::VectorData{N}) where {NX,NY,N}
    fill!(q_cache,0.0)
    grad!(q_cache,ϕ)
    normal_interpolate!(vn,q_cache,nrm,Ef,sv_cache)
end

"""
    surface_grad!(τ::VectorData,v::Edges{Primal},cache::SurfaceCache)

The operation ``G_s = n \\cdot I_f (G v + (G v)^T)``, which maps grid vector data `v` (like
velocity) to vector surface data `τ` (like traction).
"""
@inline surface_grad!(vn::VectorData,ϕ::Edges{Primal},cache::SurfaceCache) = surface_grad!(vn,ϕ,cache.nrm,cache.E,cache.gv_cache,cache.gv2_cache,cache.sv_cache)

function surface_grad!(vn::VectorData{N},ϕ::Edges{Primal,NX,NY},nrm::VectorData{N},Ef::InterpolationMatrix,gv_cache::EdgeGradient{Primal,Dual,NX,NY},gv2_cache::EdgeGradient{Primal,Dual,NX,NY},sv_cache::TensorData{N}) where {NX,NY,N}
    fill!(gv_cache,0.0)
    grad!(gv_cache,ϕ)
    normal_interpolate!(vn,gv_cache,nrm,Ef,gv2_cache,sv_cache)
end


"""
    CLinvCT(cache::SurfaceCache[;scale=1.0])

Construct the square matrix ``-C_s L^{-1}C_s^T``, which maps data of type `ScalarData`
to data of the same type. The operators `C_s` and `C_s^T` correspond to `surface_curl!`
and `L` is the grid Laplacian.
"""
function CLinvCT(cache::SurfaceCache{N};scale=1.0) where {N}
    @unpack L, ss_cache, gn_cache = cache

    len = length(ss_cache)
    A = Matrix{eltype(ss_cache)}(undef,len,len)
    fill!(ss_cache,0.0)

    for col in 1:len
        ss_cache[col] = 1.0
        fill!(gn_cache,0.0)
        surface_curl!(gn_cache,ss_cache,cache)

        gn_cache .= -(L\gn_cache);
        surface_curl!(ss_cache,gn_cache,cache)

        A[:,col] = scale*ss_cache
        fill!(ss_cache,0.0)
    end

    return A

end

"""
    GLinvD(cache::SurfaceCache[;scale=1.0])

Construct the square matrix ``G_s L^{-1}D_s``, which maps data of type `ScalarData`
to data of the same type. The operators `G_s` and `D_s` correspond to `surface_grad!`
and `surface_divergence!`, and `L` is the grid Laplacian.
"""
function GLinvD(cache::SurfaceCache{N};scale=1.0) where {N}
    @unpack L, ss_cache, gc_cache = cache

    len = length(ss_cache)
    A = Matrix{eltype(ss_cache)}(undef,len,len)
    fill!(ss_cache,0.0)

    for col in 1:len
        ss_cache[col] = 1.0
        fill!(gc_cache,0.0)
        surface_divergence!(gc_cache,ss_cache,cache)

        gc_cache .= L\gc_cache;
        surface_grad!(ss_cache,gc_cache,cache)

        A[:,col] .= scale*ss_cache
        fill!(ss_cache,0.0)
    end

    return A

end

"""
    nRTRn(cache::SurfaceCache[;scale=1.0])

Construct the square matrix ``n\\cdot I_f R_f n \\circ``, which maps data of type `ScalarData`
to data of the same type. The operators `I_f` and `R_f` correspond to the interpolation
and regularization matrices.
"""
function nRTRn(cache::SurfaceCache{N};scale=1.0) where {N}
    @unpack ss_cache, gv_cache = cache

    len = length(ss_cache)
    A = Matrix{eltype(ss_cache)}(undef,len,len)
    fill!(ss_cache,0.0)

    for col in 1:len
        ss_cache[col] = 1.0
        fill!(gv_cache,0.0)
        regularize_normal!(gv_cache,ss_cache,cache)
        normal_interpolate!(ss_cache,gv_cache,cache)

        A[:,col] .= scale*ss_cache
        fill!(ss_cache,0.0)
    end

    return A

end
