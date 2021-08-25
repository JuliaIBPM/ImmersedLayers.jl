"""
    create_CLinvCT(cache::BasicILMCache[;scale=1.0])

Using the provided cache `cache`, construct the square matrix ``-C_s L^{-1}C_s^T``, which maps data of type `ScalarData`
to data of the same type. The operators `C_s` and `C_s^T` correspond to [`surface_curl!`](@ref)
and `L` is the grid Laplacian. The optional keyword `scale` multiplies the
matrix by the designated value.
"""
function create_CLinvCT(cache::BasicILMCache{N};scale=1.0) where {N}
    @unpack L, sdata_cache, gcurl_cache = cache

    len = length(sdata_cache)
    A = Matrix{eltype(sdata_cache)}(undef,len,len)
    fill!(sdata_cache,0.0)

    for col in 1:len
        sdata_cache[col] = 1.0
        fill!(gcurl_cache,0.0)
        surface_curl!(gcurl_cache,sdata_cache,cache)

        inverse_laplacian!(gcurl_cache,cache)
        gcurl_cache .*= -1
        surface_curl!(sdata_cache,gcurl_cache,cache)

        A[:,col] = scale*sdata_cache
        fill!(sdata_cache,0.0)
    end

    return A

end

"""
    create_GLinvD(cache::BasicILMCache[;scale=1.0])

Using the provided cache `cache`, construct the square matrix ``G_s L^{-1}D_s``, which maps data of type `ScalarData`
to data of the same type. The operators `G_s` and `D_s` correspond to [`surface_grad!`](@ref)
and [`surface_divergence!`](@ref), and `L` is the grid Laplacian. The optional keyword `scale` multiplies the
matrix by the designated value.
"""
function create_GLinvD(cache::BasicILMCache{N};scale=1.0) where {N}
    @unpack L, sdata_cache, gdata_cache = cache

    len = length(sdata_cache)
    A = Matrix{eltype(sdata_cache)}(undef,len,len)
    fill!(sdata_cache,0.0)

    for col in 1:len
        sdata_cache[col] = 1.0
        fill!(gdata_cache,0.0)
        surface_divergence!(gdata_cache,sdata_cache,cache)
        inverse_laplacian!(gdata_cache,cache)
        surface_grad!(sdata_cache,gdata_cache,cache)

        A[:,col] .= scale*sdata_cache
        fill!(sdata_cache,0.0)
    end

    return A

end

"""
    create_nRTRn(cache::BasicILMCache[;scale=1.0])

Using the provided cache `cache`, construct the square matrix ``n\\cdot R_f^T R_f n \\circ``, which maps data of type `ScalarData`
to data of the same type. The operators `R_f^T` and `R_f` correspond to the interpolation
and regularization matrices. The optional keyword `scale` multiplies the
matrix by the designated value.
"""
function create_nRTRn(cache::BasicILMCache{N};scale=1.0) where {N}
    @unpack sdata_cache, gsnorm_cache = cache

    len = length(sdata_cache)
    A = Matrix{eltype(sdata_cache)}(undef,len,len)
    fill!(sdata_cache,0.0)

    for col in 1:len
        sdata_cache[col] = 1.0
        fill!(gsnorm_cache,0.0)
        regularize_normal!(gsnorm_cache,sdata_cache,cache)
        normal_interpolate!(sdata_cache,gsnorm_cache,cache)

        A[:,col] .= scale*sdata_cache
        fill!(sdata_cache,0.0)
    end

    return A

end
