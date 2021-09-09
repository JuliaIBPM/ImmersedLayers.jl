```@meta
EditURL = "<unknown>/literate/multbodies.jl"
```

# Multiple bodies

```@meta
CurrentModule = ImmersedLayers
```

Under the hood, the cache uses the concept of a `Body` to perform certain
calculations, like normal vectors and surface panel areas, which may
specialize depending on the type of body shape. Most immersed layer
operations do not depend on whether there is one or more bodies; rather,
they only depend on the discrete points, and their associated normals and areas.
However, some post-processing operations, like surface integrals, do
depend on distinguishing one body from another. For this reason, the
cache stores points in a `BodyList`, and several operations can exploit this.

````@example multbodies
using ImmersedLayers
using Plots
using LinearAlgebra
````

For the demonstration, we use the same grid.

````@example multbodies
Δx = 0.01
Lx = 4.0
xlim = (-Lx/2,Lx/2)
ylim = (-Lx/2,Lx/2)
g = PhysicalGrid(xlim,ylim,Δx)
````

We will create a 2 x 2 array of circles, each of radius 0.5, centered at $(1,1)$, $(1,-1)$,
$(-1,1)$, $(-1,-1)$.

````@example multbodies
RadC = 0.5
Δs = 1.4*cellsize(g)
body = Circle(RadC,Δs)
````

We set up the body list by pushing copies of the same body onto the list.
(We use `deepcopy` to ensure that these are copies, rather than pointers
to the same body.)

````@example multbodies
bl = BodyList()
push!(bl,deepcopy(body))
push!(bl,deepcopy(body))
push!(bl,deepcopy(body))
push!(bl,deepcopy(body))
````

Now we move them into position. We also use a `RigidTransform` for each,
which we also assemble into a list. (The `!` is for convenience, using Julia
convention, to remind us that each transform operates in-place on the body.)

````@example multbodies
t1! = RigidTransform((1.0,1.0),0.0)
t2! = RigidTransform((1.0,-1.0),0.0)
t3! = RigidTransform((-1.0,1.0),0.0)
t4! = RigidTransform((-1.0,-1.0),0.0)
tl! = RigidTransformList([t1!,t2!,t3!,t4!])
nothing #hide
````

Finally, we apply the transform. We can apply the transform list directly
to the body list:

````@example multbodies
tl!(bl)
````

Now we can create the cache, and inspect it by plotting

````@example multbodies
cache = SurfaceScalarCache(bl,g,scaling=GridScaling)
plot(cache,xlims=(-2,2),ylims=(-2,2))
````

We can now perform operations on data that exploit the division into
distinct bodies. For example, let's compute the integral of $\mathbf{x}\cdot\mathbf{n}$,
for body 3. For any of the bodies, this integral should be approximately equal to the
area enclosed by the body (or volume in 3-d), multiplied by 2 (or 3 in 3-d).
For a circle of radius $1/2$, this area is $\pi/4$, so we expect the result
to be nearly $\pi/2$. We use the `pointwise_dot` operation in
`CartesianGrids.jl` to perform the dot product at each point.

````@example multbodies
pts = points(cache)
nrm = normals(cache)
integrate(pointwise_dot(pts,nrm),cache,3)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

