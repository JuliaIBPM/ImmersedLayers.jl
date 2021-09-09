# Utilities

```@meta
CurrentModule = ImmersedLayers
```

## Surface point utilities

```@docs
areas(::Body)
normals(::Body)
points(::Body)
dot(::ScalarData{N},::ScalarData{N},::ScalarData{N}) where {N}
norm(::PointData{N},::ScalarData{N}) where {N}
integrate(::ScalarData{N},::ScalarData{N}) where {N}
ones(::ScalarData)
```
## Grid utilities

```@docs
dot(::GridData{NX,NY},::GridData{NX,NY},::PhysicalGrid) where {NX,NY}
norm(::GridData,::PhysicalGrid)
```
