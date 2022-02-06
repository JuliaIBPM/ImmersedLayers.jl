# Boundary condition functions

for f in [:prescribed_surface_jump!,:prescribed_surface_average!]
  @eval function $f(dfb,t::Real,sys::ILMSystem)
          @unpack base_cache, bc, phys_params = sys
          $f(dfb,t,base_cache,bc,phys_params)
        end
  @eval function $f(dfb,sys::ILMSystem)
          @unpack base_cache, bc, phys_params = sys
          $f(dfb,base_cache,bc,phys_params)
        end
end

function prescribed_surface_jump!(dfb,t::Real,base_cache::BasicILMCache{N},bc,phys_params) where N
    dfb .= bc["exterior"](t,base_cache,phys_params)
    dfb .-= bc["interior"](t,base_cache,phys_params)
    return dfb
end

function prescribed_surface_jump!(dfb,base_cache::BasicILMCache{N},bc,phys_params) where N
    dfb .= bc["exterior"](base_cache,phys_params)
    dfb .-= bc["interior"](base_cache,phys_params)
    return dfb
end

function prescribed_surface_jump!(dfb,t::Real,base_cache::BasicILMCache{0},bc,phys_params)
    fill!(dfb,0.0)
    return dfb
end

function prescribed_surface_jump!(dfb,base_cache::BasicILMCache{0},bc,phys_params)
    fill!(dfb,0.0)
    return dfb
end

function prescribed_surface_average!(fb,t::Real,base_cache::BasicILMCache{N},bc,phys_params) where N
    fb .= 0.5*bc["exterior"](t,base_cache,phys_params)
    fb .+= 0.5*bc["interior"](t,base_cache,phys_params)
    return fb
end

function prescribed_surface_average!(fb,t::Real,base_cache::BasicILMCache{0},bc,phys_params)
    fill!(fb,0.0)
    return fb
end

function prescribed_surface_average!(fb,base_cache::BasicILMCache{N},bc,phys_params) where N
    fb .= 0.5*bc["exterior"](base_cache,phys_params)
    fb .+= 0.5*bc["interior"](base_cache,phys_params)
    return fb
end

function prescribed_surface_average!(fb,base_cache::BasicILMCache{0},bc,phys_params)
    fill!(fb,0.0)
    return fb
end
