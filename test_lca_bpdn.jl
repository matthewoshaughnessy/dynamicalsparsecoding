# Compares three implementations of BPDN:
#   1. standard optimization-based method (using Convex.jl)
#   2. BPDN LCA using finite differences
#   3. BPDN LCA using DynamicalSystems.jl (discrete time)
#   4. BPDN LCA using DynamicalSystems.jl (continuous time)



using Random
using LinearAlgebra
using Convex, SCS
using DynamicalSystems
using Plots



# --- generate measurements ---
m = 32;
n = 64;
s = 4;
Random.seed!(0);
x = zeros(n);
x[[5,18,29,36]] = ones(s);
rerr(xhat) = norm(x-xhat) / norm(x);
Φ = randn(m,n);
Φ = Φ ./ sqrt.(sum(Φ.^2,dims=1));
y = Φ*x;
b = Φ'*y;
G = Φ'*Φ;
λ = 0.01;
τ = 1e-2;
nτ = 1000;



# --- solve with BPDN ---
xhat = Variable(n);
problem = minimize(1/2*sumsquares(y-Φ*xhat)+λ*sum(abs(xhat)));
print("Solving BPDN with Convex...");
solve!(problem,SCSSolver(),verbose=false);
xhat_bpdn = evaluate(xhat);
println("done! rerr=$(rerr(xhat_bpdn))");



# --- solve with LCA (discrete-time) ---
tt = [0 : τ : nτ*τ;];
nt = length(tt);
T(u) = u - λ*sign(u);
du(u) = 1/τ*(b-u-(G-I)*T.(u));
u_fd = zeros(n,nt);
x_lca_fd = zeros(n,nt);
print("Solving LCA by finite differences...");
for i in 2:nt
    x_lca_fd[:,i] = T.(u_fd[:,i-1]);
    u_fd[:,i] = u_fd[:,i-1] + du(u_fd[:,i-1]);
end
xhat_lca_fd = x_lca_fd[:,end];
errs_fd = [rerr(x_lca_fd[:,i]) for i in 1:nt];
println("done! rerr=$(rerr(xhat_lca_fd))");



# --- solve with LCA (discrete-time, using DynamicalSystems.jl) ---
T(u) = u - λ*sign(u);
lca_eom_d(u,p,t) = SVector{n}(u + 1/τ*(b-u-(G-I)*T.(u)));
u0 = zeros(n);
print("Solving LCA with JuliaDynamics (discrete)...");
lca_d = DiscreteDynamicalSystem(lca_eom_d, u0, nothing);
u_d_dataset = trajectory(lca_d, nτ);
u_d = zeros(n,size(u_d_dataset,1));
for i ∈ 1:n
    u_d[i,:] = u_d_dataset[:,i];
end
x_lca_d = T.(u_d);
xhat_lca_d = x_lca_d[:,end];
errs_d = [rerr(x_lca_d[:,i]) for i in 1:size(x_lca_d,2)];
println("done! rerr=$(rerr(xhat_lca_d))");



# solve with LCA (continuous-time, using DynamicalSystems.jl)
print("Solving LCA with JuliaDynamics (continuous)...");
T(u) = (u .- λ*sign.(u)) .* (u .>= λ);
lca_eom_c(u,p,t) = SVector{n}(1/τ*(b-u-(G-I)*T.(u)));
lca = ContinuousDynamicalSystem(lca_eom_c, zeros(n), nothing);
u_c = Matrix(trajectory(lca, 10))';
x_lca_c = T.(u_c);
xhat_lca_c = x_lca_c[:,end];
errs_c = [rerr(x_lca_c[:,i]) for i in 1:size(u_c,2)];
println("done! rerr=$(rerr(xhat_lca_c))\n");



# plot
p = plot(errs_fd, label="Discrete (finite differences)");
plot!(errs_d, label="Discrete (DynamicalSystems.jl, discrete)");
plot!(errs_c, label="Discrete (DynamicalSystems.jl, continuous)");
display(p);