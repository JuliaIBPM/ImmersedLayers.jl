#using Compat.Test
#using Compat

using ImmersedLayers
using Test
##using TestSetExtensions


#@test isempty(detect_ambiguities(ViscousFlow))
include("layers.jl")


#@testset ExtendedTestSet "All tests" begin
#    @includetests ARGS
#end

#if isempty(ARGS)
#    include("../docs/make.jl")
#end
