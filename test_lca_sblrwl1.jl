"""
Compares four iterative and LCA implementations of SBL:
  1. EM iterations
  2. RWL1 iterations
  3. Iterative LCA
  4. Dynamic LCA
Generates:
  - figs/lca_sbl_implementations.pdf
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
s = 12;
Random.seed!(3);
x = zeros(n);
x[rand(1:n,s)] = ones(s);
rerr(xhat) = norm(x-xhat) / norm(x);
Φ = randn(m,n);
Φ = Φ ./ sqrt.(sum(Φ.^2,dims=1));
y = Φ*x;
b = Φ'*y;
G = Φ'*Φ;
λ = 0.01;
ε = 0.01;
τ = 1e-2;
nτ = 1000;
nrw = 3;



# --- EM SBL ---
include("sbl.jl");
x_em, _, cost_em, γhist_em = sbl_em(y, Φ, σ2init = λ, updateσ2 = false, τ = Inf,
  maxiter = 10000, debug = true, verbose = false);
errs_em = rerr(x_em);



# --- iterative RWL1 (non-separable, SBL formulation) ---
include("sbl.jl");
x_iter, X_iter, W_iter, cost_iter = sbl_rwl1_simple(y, Φ, λ, τ = 0,
  maxiter = nrw*20, debug = true, verbose = true);
errs_iter = [rerr(X_iter[:,i]) for i in 1:size(X_iter,2)]



# --- Iterative LCA (non-separable, SBL formulation) ---
include("lca.jl");
Tw(u,w) = (u .- λ*w.*sign.(u)) .* (u .>= λ*w);
W_ilca = zeros(n,nrw+1);
W_ilca[:,1] = ones(n);
X_ilca = zeros(n,nτ,nrw);
γ_ilca = zeros(n,nτ,nrw);
cost_ilca = zeros(nτ,nrw);
for i ∈ 1:nrw
    # update x
    x0 = (i==1) ? zeros(n) : X_ilca[:,end,i-1];
    X_ilca[:,:,i] = wlca(b, G, Tw, τ, (nτ-1)*τ, x0, W_ilca[:,i]);
    # update w
    γ = abs.(2*X_ilca[:,end,i]) ./ W_ilca[:,i];
    W_ilca[:,i+1] = 2*sqrt.(diag(Φ'*inv(λ*I+Φ*Diagonal(γ)*Φ')*Φ));
    for j ∈ 1:nτ
      γ_ilca[:,j,i] = abs.(2*X_ilca[:,j,i]) ./ W_ilca[:,i]
      C = λ*I + Φ*Diagonal(γ_ilca[:,j,i])*Φ';
      cost_ilca[j,i] = log(det(C)) + y'*inv(C)*y;
    end
end
X_ilca = reshape(X_ilca, (n, nτ*nrw));
cost_ilca = cost_ilca[:];
errs_ilca = [rerr(X_ilca[:,i,j]) for i ∈ 1:size(X_ilca,2), j ∈ 1:size(X_ilca,3)]



# --- dynamic LCA (non-separable, using continuous time DynamicalSystems.jl) ---
include("lca.jl");
τu = τ;
τw = τ;
X_dlca, W_dlca = dsblrwl1lca(b, Φ, λ, τu, τw, (nτ-1)*τ*nrw, zeros(n), ones(n));
errs_dlca = [rerr(X_dlca[:,i]) for i in 1:size(W_dlca,2)];
cost_dlca = zeros(size(X_dlca,2));
γ_dlca = 2*abs.(X_dlca)./W_dlca;
for i ∈ 1:length(cost_dlca)
  C = λ*I + Φ*Diagonal(γ_dlca[:,i])*Φ';
  cost_dlca[i] = log(det(C)) + y'*inv(C)*y;
end



# --- plot convergence of each algorithm ---
theme(:ggplot2)
p = plot(log10.(cost_ilca.+100), 
  label=@sprintf("Iterative LCA [%.4f]",cost_ilca[end]));
plot!(p, log10.(cost_dlca.+100),
  label=@sprintf("Dynamic LCA [%.4f]",cost_dlca[end]));
plot!(p, [1,nrw*nτ], log10.((cost_em[end].+100)*[1,1]),
  label=@sprintf("SBL-EM (at convergence) [%.4f]",cost_em[end]), linestyle=:dash);
plot!(p, [1,nrw*nτ], log10.((cost_iter[end].+100)*[1,1]), 
  label=@sprintf("SBL-RWL1 (at convergence) [%.4f]",cost_iter[end]), linestyle=:dashdot);
plot(p, linewidth=2, xlab="Time constants", ylab="Scaled cost (log scale)")
plot(p, title="SBL implementations converge to nearly the same final cost")
plot(p, ylim=[1.2,1.9]);
p
savefig("./figs/lca_sbl_implementations.pdf")



# --- dynamic LCA (non-separable, implemented with first-differences)
#=
nt = 10000;
τu = 10;
τw = 1;
Δ = 1e-2;
u_lns = zeros(n,nt);
x_lns = zeros(n,nt);
w_lns = zeros(n,nt);
du_lns = zeros(n,nt);
dw_lns = zeros(n,nt);
u_lns[:,1] = zeros(n);
w_lns[:,1] = ones(n);
Tw(u,w) = (u .- λ*w.*sign.(u)) .* (u .>= λ*w);
for i in 1:nt-1
    # u[:,i+1] = u[:,i] + du[:,i]
    i % 100 == 0 ? println("Evolving from t = $(i) to $(i+1)") : nothing;
    du_lns[:,i] = b .- u_lns[:,i] .- (G-I)*x_lns[:,i];
    dw_lns[:,i] = w_lns[:,i].^-1 .- diag(Φ'*inv(λ*I+Φ*Diagonal(abs.(x_lns[:,i])./w_lns[:,i])*Φ')*Φ).^-1;
    #dw_lns[:,i] = w_lns[:,i].^-1 .- (abs.(x_lns[:,i]) .+ ε);
    u_lns[:,i+1] = u_lns[:,i] + Δ/τu*du_lns[:,i];
    w_lns[:,i+1] = w_lns[:,i] + Δ/τw*dw_lns[:,i];
    x_lns[:,i+1] = Tw(u_lns[:,i+1],w_lns[:,i+1]);
end
X_lns = x_lns;
errs_lns = [rerr(X_lns[:,i]) for i in 1:size(X_lns,2)]
=#



# --- dynamic LCA (non-separable, using first-differences function) ---
#=
include("lca.jl");
τu = 1e12;
τw = 1e3;
x_dsblrwl1lca, X_dsblrwl1lca, W_dsblrwl1lca, U_dsblrwl1lca, dw, du = dsblrwl1lcafd(
  b, Φ, λ, τu, τw, 10, ones(n), 1e-4*ones(n))
errs_dsblrwl1lca = [rerr(X_dsblrwl1lca[:,i]) for i in 1:size(X_dsblrwl1lca,2)];
=#



# --- plot ---
#=
theme(:ggplot2)
p = plot(log.(cost_dlca.+100), label = "LCA (dynamic reweighting)", lw = 2,
        xlab = "Time constants", ylab = "log10(cost) [scaled]",
        title = "RWL1-SBL-LCA: cost as LCA dynamical system evolves");
plot!(p, log.(cost_ilca.+100), label = "LCA (iterative reweighting)", lw = 2,
    line = :dash);
for i in 1:nrw
    plot!(p, [(i-1)*nτ; i*nτ], log(cost_iter[i].+100)*ones(2),
        label = "Iterative: reweight $(i)", line = :dot, lw = 2);
end
p
#savefig("./figs/lca_sblrwl1_dyn_vs_iter.pdf")
=#