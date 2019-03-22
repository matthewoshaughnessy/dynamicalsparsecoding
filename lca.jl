# Contains functions that implement variants of the LCA

using DynamicalSystems

function lca(b, G, T, τ, tf, u0)
# lca : implements a standard LCA with generic thresholding function
# Inputs:
#   b  : Φ^T*y
#   G  : Φ^T*Φ
#   T  : thresholding function, x = T(u)
#   τ  : LCA time constant (used in ODE definition of LCA, not ODE solver)
#   tf : amount of time to simulate
#   u0 : initial state for u
# Outputs:
#   x  : trajectory of x, n x nt

    eom(u,p,t) = SVector{n}(1/τ*(b-u-(G-I)*T(u)));
    lca = ContinuousDynamicalSystem(eom, u0, nothing);
    u = Matrix(trajectory(lca, tf))';
    x = T(u);
    
    return x

end

function wlca(b, G, T, τ, tf, u0, w)
# wlca : implements a weighted LCA with fixed weights and generic thresholding function
# Inputs:
#   b  : Φ^T*y
#   G  : Φ^T*Φ
#   T  : thresholding function, x = T(u)
#   τ  : LCA time constant (used in ODE definition of LCA, not ODE solver)
#   tf : amount of time to simulate
#   u0 : initial state for u
#   w  : fixed weights, n x 1
# Outputs:
#   x  : trajectory of x, n x nt

    eom(u,p,t) = SVector{n}(1/τ*(b-u-(G-I)*T(u,p)));
    lca = ContinuousDynamicalSystem(eom, u0, w);
    u = Matrix(trajectory(lca, tf))';
    x = T(u,w);
    
    return x

end

function drwl1lca(b, G, T, τu, τw, tf, u0, w0, ε = 0.01)
# drwl1lca : implements a dynamic rwl1 LCA, with generic thresholding function
# Inputs:
#   b  : Φ^T*y
#   G  : Φ^T*Φ
#   T  : thresholding function, x = T(u)
#   τu : LCA time constant for state u (used in ODE definition of LCA, not ODE solver)
#   τw : LCA time constant for weights w (used in ODE definition of LCA, not ODE solver)
#   tf : amount of time to simulate
#   u0 : initial state for u
#   w0 : initial state for w
#   ε  : constant used in reweighting (default: 0.01)
# Outputs:
#   x  : trajectory of x, n x nt
#   w  : trajectory of w, n x nt

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