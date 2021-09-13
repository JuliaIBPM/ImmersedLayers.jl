"""
    create_RTLinvR(cache::BasicILMCache[;scale=1.0])

Using the provided cache `cache`, construct the square matrix ``-R^T L^{-1}R``, which maps data of type `ScalarData`
to data of the same type. The operators `R^T` and `R` correspond to [`interpolate!`](@ref) and
[`regularize!`](@ref) and `L^{-1}` is the inverse of the grid Laplacian. The optional keyword `scale` multiplies the
matrix by the designated value.
"""
function create_RTLinvR(cache::BasicILMCache{N};scale=1.0) where {N}
    @unpack L, sdata_cache, gdata_cache = cache

    len = length(sdata_cache)
    A = Matrix{eltype(sdata_cache)}(undef,len,len)
    fill!(sdata_cache,0.0)

    for col in 1:len
        sdata_cache[col] = 1.0
        fill!(gdata_cache,0.0)

        regularize!(gdata_cache,sdata_cache,cache)
        inverse_laplacian!(gdata_cache,cache)
        interpolate!(sdata_cache,gdata_cache,cache)

        A[:,col] = -scale*sdata_cache
        fill!(sdata_cache,0.0)
    end

    return A

end

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
        surface_curl!(sdata_cache,gcurl_cache,cache)

        A[:,col] = -scale*sdata_cache
        fill!(sdata_cache,0.0)
    end

    return A

end

"""
    create_GLinvD(cache::BasicILMCache[;scale=1.0])

Using the provided cache `cache`, construct the square matrix ``-G_s L^{-1}D_s``, which maps data of type `ScalarData`
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

        A[:,col] .= -scale*sdata_cache
        fill!(sdata_cache,0.0)
    end

    return A

end

"""
    create_GLinvD_cross(cache::BasicILMCache[;scale=1.0])

Using the provided cache `cache`, construct the square matrix ``-\\hat{G}_s L^{-1}\\hat{D}_s``, which maps data of type `ScalarData`
to data of the same type. The operators `G_s` and `D_s` correspond to [`surface_grad_cross!`](@ref)
and [`surface_divergence_cross!`](@ref), and `L` is the grid Laplacian. The optional keyword `scale` multiplies the
matrix by the designated value.
"""
function create_GLinvD_cross(cache::BasicILMCache{N};scale=1.0) where {N}
    @unpack L, sdata_cache, gdata_cache = cache

    len = length(sdata_cache)
    A = Matrix{eltype(sdata_cache)}(undef,len,len)
    fill!(sdata_cache,0.0)

    for col in 1:len
        sdata_cache[col] = 1.0
        fill!(gdata_cache,0.0)
        surface_divergence_cross!(gdata_cache,sdata_cache,cache)
        inverse_laplacian!(gdata_cache,cache)
        surface_grad_cross!(sdata_cache,gdata_cache,cache)

        A[:,col] .= -scale*sdata_cache
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

"""
    create_surface_filter(cache::BasicILMCache)

Create a surface filtering matrix operator ``\\tilde{R}^T R``, where ``\\tilde{R}^T``
represents a modified version of the
interpolation operator. The resulting matrix can be applied
to surface data to filter out high-frequency components.
"""
function create_surface_filter(cache::BasicILMCache{N,SCA}) where {N,SCA}
    @unpack R, gdata_cache, sdata_cache, g = cache
    regfilt = _get_regularization(points(cache),areas(cache),g,
                                  _ddf_type(cache),SCA,filter=true)
    Ef = InterpolationMatrix(regfilt,gdata_cache,sdata_cache)
    C = zeros(N,N)
    return mul!(C,Ef,R)

end
