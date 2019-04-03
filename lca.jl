# Contains functions that implement variants of the LCA

using DynamicalSystems

"""
Implements a standard LCA with generic thresholding function
Inputs:
  `b`  : Φ^T*y
  `G`  : Φ^T*Φ
  `T`  : thresholding function, x = T(u)
  `τ`  : LCA time constant (used in ODE definition of LCA, not ODE solver)
  `tf` : amount of time to simulate
  `u0` : initial state for u
Outputs:
  `x`  : trajectory of x, n x nt
"""
function lca(b, G, T, τ, tf, u0)

    eom(u,p,t) = SVector{n}(1/τ*(b-u-(G-I)*T(u)));
    lca = ContinuousDynamicalSystem(eom, u0, nothing);
    u = Matrix(trajectory(lca, tf))';
    x = T(u);
    
    return x

end

function lca_j(b, G, T, τ, tf, u0)
# lca : implements a standard LCA with generic thresholding function
#       with Jacobian specified
# Inputs:
#   b  : Φ^T*y
#   G  : Φ^T*Φ
#   T  : thresholding function, x = T(u)
#   τ  : LCA time constant (used in ODE definition of LCA, not ODE solver)
#   tf : amount of time to simulate
#   u0 : initial state for u
# Outputs:
#   x  : trajectory of x, n x nt
# TODO
    
    #function eom_lca_j!(du, u, p, t)
    #    du = 1/τ*(b-u-(G-I)*T(u))
    #    return
    #end
    eom(u,p,t) = SVector{n}(1/τ*(b-u-(G-I)*T(u)));

    #function eom_jac(J, x, p, n)
    #    J = Diagonal(ones(2))
    #    return
    #end

    lca = ContinuousDynamicalSystem(eom, u0, nothing);
    u = Matrix(trajectory(lca, tf))';
    x = T(u);
    
    return x, u

end

"""
wlca : implements a weighted LCA with fixed weights and generic thresholding function
Inputs:
  `b`  : Φ^T*y
  `G`  : Φ^T*Φ
  `T`  : thresholding function, x = T(u)
  `τ`  : LCA time constant (used in ODE definition of LCA, not ODE solver)
  `tf` : amount of time to simulate
  `u0` : initial state for u
  `w`  : fixed weights, n x 1
Outputs:
  `x`  : trajectory of x, n x nt
"""
function wlca(b, G, T, τ, tf, u0, w)

    eom(u,p,t) = SVector{n}(1/τ*(b-u-(G-I)*T(u,p)));
    lca = ContinuousDynamicalSystem(eom, u0, w);
    u = Matrix(trajectory(lca, tf))';
    x = T(u,w);
    
    return x

end

"""
Implements a dynamic rwl1 LCA, with generic thresholding function
Inputs:
  `b`  : Φ^T*y
  `G`  : Φ^T*Φ
  `T`  : thresholding function, x = T(u)
  `τu` : LCA time constant for state u (used in ODE definition of LCA, not ODE solver)
  `τw` : LCA time constant for weights w (used in ODE definition of LCA, not ODE solver)
  `tf` : amount of time to simulate
  `u0` : initial state for u
  `w0` : initial state for w
  `ε`  : constant used in reweighting (default: 0.01)
Outputs:
  `x`  : trajectory of x, n x nt
  `w`  : trajectory of w, n x nt
"""
function drwl1lca(b, G, T, τu, τw, tf, u0, w0, ε = 0.01)

    n = size(G,1);
    function eom(uu,p,t) # uu : combined state [u; w]
        u = uu[1:n];
        w = uu[n+1:end];
        du = 1/τu*(b-u-(G-I)*T(u,w));
        dw = 1/τw*(w.^-1 .- (abs.(T(u,w)) .+ ε));
        SVector{2*n}(vcat(du,dw));
    end
    uu0 = [u0; w0];
    lca = ContinuousDynamicalSystem(eom, uu0, nothing);
    uu = Matrix(trajectory(lca,tf))';
    u = uu[1:n,:];
    w = uu[n+1:end,:];
    x = T(u,w);
    return x, w

end

"""
Implements a dynamic rwl1 LCA, with SBL-RWL1 thresholding function, using first-differences
Inputs:
  `b`  : Φ^T*y
  `Φ`  : dictionary
  `λ`  : SBL regularization parameter
  `τu` : LCA time constant for state u (used in ODE definition of LCA, not ODE solver)
  `τw` : LCA time constant for weights w (used in ODE definition of LCA, not ODE solver)
  `nt` : number of time steps to simulate
  `u0` : initial state for u
  `w0` : initial state for w
Outputs:
  `x`  : trajectory of x, n x nt
  `w`  : trajectory of w, n x nt
"""
function dsblrwl1lcafd(b, Φ, λ, τu, τw, nt, u0, w0)

    n = size(Φ,2)
    G = Φ'*Φ
    Tw(u,w) = (u .- λ*w.*sign.(u)) .* (u .>= λ*w)
    function dudw(u,w)
        du = 1/τu*(b-u-(G-I)*Tw(u,w))
        dw = 1/τw*(w .- (2*sqrt.(diag(Φ'*inv(λ*I+Φ*Diagonal(abs.(Tw(u,w))./w)*Φ')*Φ))))
        return du, dw
    end

    u = zeros(n,nt)
    w = zeros(n,nt)
    duhist = zeros(n,nt) # DEBUG
    dwhist = zeros(n,nt) # DEBUG
    u[:,1] = u0
    w[:,1] = w0

    for i in 2:nt
        du, dw = dudw(u[:,i],w[:,i])
        duhist[:,i-1] = du # DEBUG
        dwhist[:,i-1] = dw # DEBUG
        u[:,i] = u[:,i-1] + du
        w[:,i] = w[:,i-1] + dw
    end

    x = zeros(n,nt)
    for i in 1:nt
        x[:,i] = Tw(u[:,i],w[:,i])
    end
    xhat = x[:,end]
    return xhat, x, w, u, dwhist, duhist

end

"""
Implements a dynamic rwl1 LCA, with SBL-RWL1 thresholding function
Inputs:
  `b`  : Φ^T*y
  `Φ`  : dictionary
  `λ`  : SBL regularization parameter
  `τu` : LCA time constant for state u (used in ODE definition of LCA, not ODE solver)
  `τw` : LCA time constant for weights w (used in ODE definition of LCA, not ODE solver)
  `tf` : amount of time to simulate
  `u0` : initial state for u
  `w0` : initial state for w
Outputs:
  `x`  : trajectory of x, n x nt
  `w`  : trajectory of w, n x nt
"""
function dsblrwl1lca(b, Φ, λ, τu, τw, tf, u0, w0)

    n = size(Φ,2)
    G = Φ'*Φ
    Tw(u,w) = (u .- λ*w.*sign.(u)) .* (u .>= λ*w)
    function eom(duu, uu, p, t) # uu : combined state [u; w]
        u = uu[1:n]
        w = uu[n+1:end]
        du = 1/τu*(b-u-(G-I)*Tw(u,w))
        dw = 1/τw*(w.^-1 .- (2*sqrt.(diag(Φ'*inv(λ*I+Φ*Diagonal(abs.(Tw(u,w))./w)*Φ')*Φ))).^-1)
        duu[1:n] = du
        duu[n+1:end] = dw
        return
    end
    uu0 = [u0; w0]
    lca = ContinuousDynamicalSystem(eom, uu0, nothing)
    #lca = DiscreteDynamicalSystem(eom, uu0, nothing)
    uu = Matrix(trajectory(lca,tf))'
    u = uu[1:n,:]
    w = uu[n+1:end,:]
    x = Tw(u,w)
    return x, w

end