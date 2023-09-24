# Utilities

```@meta
CurrentModule = ImmersedLayers
```

## Surface point utilities on bodies and body lists

```@docs
areas(::Union{Body,BodyList})
normals(::Union{Body,BodyList})
points(::Union{Body,BodyList})
arcs(::Union{Body,BodyList})
ones(::PointData{N}) where {N}
ones(::Union{TensorData,VectorData},::Integer)
dot(::ScalarData{N},::ScalarData{N},::ScalarData{N}) where {N}
dot(::ScalarData{N},::ScalarData{N},::ScalarData{N},::BodyList,::Int) where {N}
norm(::PointData{N},::ScalarData{N}) where {N}
norm(::PointData{N},::ScalarData{N},::BodyList,::Int) where {N}
integrate(::ScalarData{N},::ScalarData{N}) where {N}
integrate(::ScalarData{N},::ScalarData{N},::BodyList,::Int) where {N}
copyto!(::PointData,::PointData,::BodyList,::Int)
copyto!(::ScalarData{N},::AbstractVector,::BodyList,::Int) where {N}
view(::VectorData,::BodyList,::Int)
```


## Grid utilities

```@docs
ones(::GridData)
ones(::Union{TensorGridData,VectorGridData},::Integer)
integrate(::ScalarGridData,::PhysicalGrid)
dot(::GridData{NX,NY},::GridData{NX,NY},::PhysicalGrid) where {NX,NY}
norm(::GridData,::PhysicalGrid)
```
