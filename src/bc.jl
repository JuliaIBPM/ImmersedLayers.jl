# Boundary condition functions

for f in [:prescribed_surface_jump!,:prescribed_surface_average!]
  @eval function $f(dfb,t::Real,sys::ILMSystem)
          @unpack base_cache, bc, phys_params, motions = sys
          $f(dfb,t,base_cache,bc,phys_params,motions)
        end
  @eval function $f(dfb,sys::ILMSystem)
          @unpack base_cache, bc, phys_params = sys
          $f(dfb,base_cache,bc,phys_params)
        end
end

function prescribed_surface_jump!(dfb,t::Real,base_cache::BasicILMCache{N},bc,phys_params,motions) where N
    dfb .= _get_function_name(bc["exterior"])(t,base_cache,phys_params,motions)
    dfb .-= _get_function_name(bc["interior"])(t,base_cache,phys_params,motions)
    return dfb
end

function prescribed_surface_jump!(dfb,base_cache::BasicILMCache{N},bc,phys_params) where N
    dfb .= _get_function_name(bc["exterior"])(base_cache,phys_params)
    dfb .-= _get_function_name(bc["interior"])(base_cache,phys_params)
    return dfb
end

function prescribed_surface_jump!(dfb,t::Real,base_cache::BasicILMCache{0},bc,phys_params,motions)
    fill!(dfb,0.0)
    return dfb
end

function prescribed_surface_jump!(dfb,base_cache::BasicILMCache{0},bc,phys_params)
    fill!(dfb,0.0)
    return dfb
end

function prescribed_surface_average!(fb,t::Real,base_cache::BasicILMCache{N},bc,phys_params,motions) where N
    fb .= 0.5*_get_function_name(bc["exterior"])(t,base_cache,phys_params,motions)
    fb .+= 0.5*_get_function_name(bc["interior"])(t,base_cache,phys_params,motions)
    return fb
end

function prescribed_surface_average!(fb,t::Real,base_cache::BasicILMCache{0},bc,phys_params,motions)
    fill!(fb,0.0)
    return fb
end

function prescribed_surface_average!(fb,base_cache::BasicILMCache{N},bc,phys_params) where N
    fb .= 0.5*_get_function_name(bc["exterior"])(base_cache,phys_params)
    fb .+= 0.5*_get_function_name(bc["interior"])(base_cache,phys_params)
    return fb
end

function prescribed_surface_average!(fb,base_cache::BasicILMCache{0},bc,phys_params)
    fill!(fb,0.0)
    return fb
end
