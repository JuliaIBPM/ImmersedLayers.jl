```@meta
EditURL = "<unknown>/literate/stokes.jl"
```

# Stokes flow

```@meta
CurrentModule = ImmersedLayers
```

Here, we'll demonstrate a solution of the Stokes equations,
with Dirichlet (i.e. no-slip) boundary conditions on a surface.
The purpose of this case is to demonstrate the use of the tools with
vector-valued data -- in this case, the fluid velocity field.

The governing equations for this problem can be written as

$$\mu \nabla^2 \omega - \nabla \times (\delta(\chi) \sigma) = \nabla\times \nabla \cdot (\delta(\chi)\mathbf{\Sigma})$$
$$\delta^T(\chi) \mathbf{v} = \overline{\mathbf{v}}_b$$
$$\mathbf{v} = \nabla \phi + \nabla\times \psi$$

where $\psi$ and $\phi$ are the solutions of

$$\nabla^2 \psi = -\omega$$
$$\nabla^2 \phi = \delta(\chi) \mathbf{n}\cdot [\mathbf{v}_b]$$

and $\mathbf{\Sigma} = \mu ([\mathbf{v}_b]\mathbf{n} + \mathbf{n} [\mathbf{v}_b])$ is a surface viscous flux tensor;
and $[\mathbf{v}_b] = \mathbf{v}_b^+ - \mathbf{v}_b^-$ and $\overline{v}_b = (\mathbf{v}_b^+ + \mathbf{v}_b^-)/2$ are
the jump and average of the surface velocities on either side of a surface.

We can discretize and combine these equations into a saddle-point form for $\psi$ and $\sigma$:

$$\begin{bmatrix}L^2 & C_s^T \\ C_s & 0 \end{bmatrix}\begin{pmatrix} \psi \\ \sigma \end{pmatrix} = \begin{pmatrix} -C D_s [\mathbf{v}_b]  \\ \overline{v}_b - G_s L^{-1} R \mathbf{n}\cdot [\mathbf{v}_b]  \end{pmatrix}$$

This saddle-point system has a form similar to the one we got

````@example stokes
using ImmersedLayers
using Plots
using LinearAlgebra
using UnPack
````

## Set up the extra cache and solve function
The problem type takes the usual basic form

````@example stokes
struct StokesFlowProblem{DT,ST} <: ImmersedLayers.AbstractVectorILMProblem{DT,ST}
   g :: PhysicalGrid
   bodies :: BodyList
   StokesFlowProblem(g::PT,bodies::BodyList;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,bodies)
   StokesFlowProblem(g::PT,body::Body;ddftype=CartesianGrids.Yang3,scaling=IndexScaling) where {PT} = new{ddftype,scaling}(g,BodyList([body]))
end
````

As with other cases, the extra cache holds additional intermediate data, as well as
the Schur complement. We don't bother creating a filtering matrix here.

````@example stokes
struct StokesFlowCache{SMT,RCT,DVT,VNT,ST,VFT,FT} <: AbstractExtraILMCache
   S :: SMT
   Rc :: RCT
   dv :: DVT
   vb :: DVT
   vprime :: DVT
   dvn :: VNT
   sstar :: ST
   vϕ :: VFT
   ϕ :: FT
end

function ImmersedLayers.prob_cache(prob::StokesFlowProblem,base_cache::BasicILMCache)
    S = create_CL2invCT(base_cache)

    dv = zeros_surface(base_cache)
    vb = zeros_surface(base_cache)
    vprime = zeros_surface(base_cache)

    dvn = ScalarData(dv)
    sstar = zeros_gridcurl(base_cache)
    vϕ = zeros_grid(base_cache)
    ϕ = Nodes(Primal,sstar)

    Rc = RegularizationMatrix(base_cache,dvn,ϕ)

    StokesFlowCache(S,Rc,dv,vb,vprime,dvn,sstar,vϕ,ϕ)
end

Δx = 0.01
Lx = 4.0
xlim = (-Lx/2,Lx/2)
ylim = (-Lx/2,Lx/2)
g = PhysicalGrid(xlim,ylim,Δx)
Δs = 1.4*cellsize(g);

function ImmersedLayers.solve(vsplus,vsminus,prob::StokesFlowProblem,sys::ILMSystem)
    @unpack extra_cache, base_cache = sys
    @unpack S, Rc, dv, vb, vprime, sstar, dvn, vϕ, ϕ  = extra_cache

    σ = zeros_surface(sys)
    s = zeros_gridcurl(sys)
    v = zeros_grid(sys)

    dv .= vsplus-vsminus
    vb .= 0.5*(vsplus + vsminus)

    # Compute ψ*
    surface_divergence!(v,dv,sys)
    curl!(sstar,v,sys)
    sstar .*= -1.0

    inverse_laplacian!(sstar,sys)
    inverse_laplacian!(sstar,sys)

    # Adjustment for jump in normal velocity
    pointwise_dot!(dvn,nrm,dv)
    regularize!(ϕ,dvn,Rc)
    inverse_laplacian!(ϕ,sys)
    grad!(vϕ,ϕ,sys)
    interpolate!(vprime,vϕ,sys)
    vprime .= vb - vprime

    # Compute surface velocity due to ψ*
    curl!(v,sstar,sys)
    interpolate!(vb,v,sys)

    # Spurious slip
    vprime .-= vb
    σ .= S\vprime

    # Correction streamfunction
    surface_curl!(s,σ,sys)
    s .*= -1.0
    inverse_laplacian!(s,sys)
    inverse_laplacian!(s,sys)

    # Correct
    s .+= sstar

    # Assemble the velocity
    curl!(v,s,sys)
    v .+= vϕ

    return v, s, σ, vprime
end

body = Circle(0.5,Δs)

prob = StokesFlowProblem(g,body,scaling=GridScaling)

sys = ImmersedLayers.__init(prob);

pts = points(sys);
nrm = normals(sys);

vsplus = zeros_surface(sys);
vsminus = zeros_surface(sys);
vsplus.u .= -nrm.v;
vsplus.v .= nrm.u;
##vsplus.u .= 1.0;
##vsplus.v .= 0.0;

vsminus.u .= 0.0;

@time v, s, σ, vprime = solve(vsplus,vsminus,prob,sys);

plot(v,sys)

plot(s,sys)

vb = zeros_surface(sys);
vb .= 0.5*(vsplus + vsminus);
vs = zeros_surface(sys);

interpolate!(vs,v,sys);

plot(vs)

plot(σ.u)

F = svd(sys.extra_cache.S);

plot(1.0./F.S,yscale=:log10)
plot!(F.S./(F.S.^2 .+ 1e-6))

σ2 = zeros_surface(sys);

Σ⁺ = Diagonal(F.S./(F.S.^2 .+ 1e-6))
S⁺ = F.Vt'*Σ⁺*F.U';
σ2 .= S⁺*vprime;

plot(σ.u)
plot!(σ2.u)

norm(sys.extra_cache.S*σ2-vprime)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

