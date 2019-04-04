using Random
using LinearAlgebra
using Convex, SCS
using DynamicalSystems
using Plots, PlotThemes



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
λ = 0.01;
maxiter = Dict("EM" => 1000, "RWL1" => 100);



# --- solve with BPDN ---
xc = Variable(n);
problem = minimize(1/2*sumsquares(y-Φ*xc)+λ*sum(abs(xc)));
solve!(problem, SCSSolver(verbose = false), verbose=false);
xhat_bpdn = evaluate(xc);
err_bpdn = rerr(xhat_bpdn)



# --- solve with SBL using EM iterations ---
include("sbl.jl");
xhat_sbl, σ2_sbl, cost_sbl = sbl_em(y, Φ, σ2init = λ, updateσ2 = false,
    maxiter = maxiter["EM"], verbose = false, debug = true);
err_sbl = rerr(xhat_sbl)



# --- solve with SBL using RWL1 iterations ---
include("sbl.jl");
xhat_sblrwl1, Xhat_sblrwl1, Γhat_sblrwl1, cost_sblrwl1 = sbl_rwl1(y, Φ, λ,
    maxiter = maxiter["RWL1"], verbose = false, debug = true);
errs_sblrwl1 = [rerr(Xhat_sblrwl1[:,i]) for i in 1:size(Xhat_sblrwl1,2)]



# --- solve with SBL using RWL1 iterations (no latent variables) ---
include("sbl.jl");
xhat_sblrwl1_simple, Xhat_sblrwl1_simple, What_sblrwl1_simple, cost_sblrwl1_simple = sbl_rwl1_simple(y, Φ, λ,
    maxiter = maxiter["RWL1"], verbose = false, debug = true);
errs_sblrwl1_simple = [rerr(Xhat_sblrwl1_simple[:,i]) for i in 1:size(Xhat_sblrwl1_simple,2)]



# --- plot SBL-RWL1 iterations ---
p = plot()
for i in 1:n
    if x[i] != 0
        plot!(p, [1; What_sblrwl1_simple[i,:]], color=RGBA(0.3,0.1,1.0,0.8), label = "")
    else
        plot!(p, [1; What_sblrwl1_simple[i,:]], label = "", xlab = "RWL1 iteration", ylab = "w_i", 
            color=RGBA(1.0,0.1,0.3,0.8))
    end
end
display(p)
savefig("./figs/sbl_rwl1.pdf")