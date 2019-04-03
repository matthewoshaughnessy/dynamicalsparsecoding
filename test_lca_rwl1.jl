"""
Compares three implementations of separable RWL1:
  1. standard iterative reweighted l1 (using Convex.jl)
  2. RWL1 implemented using a sequence of weighted-BPDN LCAs
  3. RWL1 implemented using a single dynamic reweighted-BPDN LCA
Generates:
  - figs/lca_rwl1_dyn_vs_iter.pdf
"""



using Random
using LinearAlgebra
using Convex, SCS
using DynamicalSystems
using Plots
using PlotThemes



# --- generate measurements ---
m = 32;
n = 64;
s = 13;
Random.seed!(0);
x = zeros(n);
x[rand(1:n,s)] = ones(s);
rerr(xhat) = norm(x-xhat) / norm(x);
Φ = randn(m,n);
Φ = Φ ./ sqrt.(sum(Φ.^2,dims=1));
y = Φ*x;
b = Φ'*y;
G = Φ'*Φ;
λ0 = 0.01;
ε = 0.01;
τ = 1e-2;
nτ = 500;
nrw = 3;



# --- iterative RWL1 (separable) ---
W_rwl1 = zeros(n,nrw+1);
W_rwl1[:,1] = ones(n);
Xhat_rwl1 = zeros(n,nrw);
for i ∈ 1:nrw
    xhat = Variable(n);
    problem = minimize( 1/2*sumsquares(y-Φ*xhat) + λ0*sum(W_rwl1[:,i].*abs(xhat)) );
    solve!(problem,SCSSolver(verbose=false),verbose=false);
    Xhat_rwl1[:,i] = evaluate(xhat);
    W_rwl1[:,i+1] = 1 ./ (abs.(Xhat_rwl1[:,i]) .+ ε);
end
errs_rwl1 = [rerr(Xhat_rwl1[:,i]) for i ∈ 1:nrw];



# --- Iterative LCA (separable) ---
include("lca.jl");
Tw(u,w) = (u .- λ0*w.*sign.(u)) .* (u .>= λ0*w);
W_ilca = zeros(n,nrw+1);
W_ilca[:,1] = ones(n);
Xhat_ilca = zeros(n,nτ,nrw);
for i ∈ 1:nrw
    x0 = (i==1) ? zeros(n) : Xhat_ilca[:,end,i-1];
    Xhat_ilca[:,:,i] = wlca(b, G, Tw, τ, (nτ-1)*τ, x0, W_ilca[:,i]);
    W_ilca[:,i+1] = 1 ./ (Xhat_ilca[:,end,i] .+ ε);
end
Xhat_ilca = reshape(Xhat_ilca, (n, nτ*nrw));
errs_ilca = [rerr(Xhat_ilca[:,i]) for i ∈ 1:size(Xhat_ilca,2)];



# --- dynamic LCA (separable) ---
include("lca.jl");
Xhat_dlca, W_dlca = drwl1lca(b, G, Tw, τ, τ, (nτ-1)*nrw*τ, zeros(n), λ0*ones(n), ε);
errs_dlca = [rerr(Xhat_dlca[:,i]) for i in 1:size(Xhat_dlca,2)];



# --- plot ---
theme(:ggplot2)
p = plot(log.(errs_dlca), label = "LCA (dynamic reweighting)", lw = 2,
        xlab = "Time constants", ylab = "log10(rMSE)",
        title = "rMSE as LCA dynamical system evolves");
plot!(p, log.(errs_ilca), label = "LCA (iterative reweighting)", lw = 2,
    line = :dash);
for i in 1:nrw
    plot!(p, [(i-1)*nτ; i*nτ], log(errs_rwl1[i])*ones(2),
        label = "Iterative RWL1: reweight $(i)", line = :dot, lw = 2);
end
p
savefig("./figs/lca_rwl1_dyn_vs_iter.pdf")