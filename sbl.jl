# Contains functions that implement variants of SBL

using LinearAlgebra
using Printf
using Convex


"""
Implements SBL using the EM updates of [1]
Inputs:
  `y`        : measurements, m x 1
  `Φ`        : dictionary, m x n
  `σ2init`   : initial value of noise variance parameter (default: 1e-3)
  `updateσ2` : update σ2 during EM procedure? (default: true)
  `τ`        : threshold on γ used for pruning (default: 1e4)
  `tol`      : convergence tolerance (default: 1e-8)
  `maxiter`  : maximum iteration count (default: 100)
  `verbose`  : print debug information at each iteration (default: false)
  `debug`    : calculate and output cost function value at each iteration (default: false)
Outputs:
  `xhat`     : SBL estimate of x (i.e., posterior mean at convergence)
Reference:
  [1]        : M Tipping, 'Sparse Bayesian learning and the relevance vector machine,'
               J Machine Learning Research, 2001.
"""
function sbl_em(y, Φ; σ2init = 1e-3, updateσ2 = true, τ = 1e8, tol = 1e-8, maxiter = 100, verbose = false, debug = false)

    # initialization
    m, n = size(Φ)
    σ2 = σ2init
    γ = ones(n)
    γsupp = trues(n)
    μ = zeros(n)
    iter = 1
    converged = false
    mackay = zeros(n)
    cost = debug ? zeros(maxiter) : nothing

    # parameters
    a = zeros(n)
    b = zeros(n)
    c = 0
    d = 0

    # perform EM iterations
    while !converged

        verbose ? print("Iteration $(iter): ") : nothing;

        # update posterior
        Σ = inv(1/σ2*Φ[:,γsupp]'*Φ[:,γsupp] + Diagonal(γ[γsupp]))
        μ[γsupp] = 1/σ2*Σ*Φ[:,γsupp]'*y
        μ[.!γsupp] = zeros(sum(.!γsupp))

        # update γ
        γ_old = γ[:]
        for i in findall(γsupp)
            ip = sum(γsupp[1:i])
            γ[i] = (1 + 2*a[i]) / (μ[i]^2 + Σ[ip,ip] + 2*b[i]);
        end

        # update σ2
        if updateσ2
            mackay[γsupp] = 1 .- γ[γsupp].*diag(Σ)
            σ2 = (norm(y-Φ[:,γsupp]*μ[γsupp])^2 + σ2*sum(mackay[γsupp]) + 2*d) / (m + 2*c)
        end

        # prune and check convergence
        γsupp_old = γsupp[:]
        γsupp = γsupp .& (abs.(γ) .<= τ)
        converged = norm(γ-γ_old) < tol || iter >= maxiter
        if verbose
            @printf("%d coefficients in model; ", Int(sum(γsupp)));
            @printf("||γ|| = %.3e, ||γ-γold|| = %.3e.\n", norm(γ), norm(γ-γ_old));
        end
        if debug
          C = λ*I + Φ*Diagonal(γ.^-1)*Φ'
          cost[iter] = log(det(C)) + y'*inv(C)*y
        end
        iter += 1

    end

    # calculate final posterior
    Σ = inv(1/σ2*Φ[:,γsupp]'*Φ[:,γsupp] + Diagonal(γ[γsupp]))
    μ[γsupp] = 1/σ2*Σ*Φ[:,γsupp]'*y
    μ[.!γsupp] = zeros(sum(.!γsupp))
    xhat = μ
    cost = debug ? cost[1:iter-1] : nothing

    return xhat, σ2, cost

end


"""
Implements SBL using the RWL1 updates of [2]
Inputs:
  `y`        : measurements, m x 1
  `Φ`        : dictionary, m x n
  `λ`        : data fidelity/sparsity tuning parameter
  `τ`        : threshold on γ used for pruning (default: 1e4)
  `tol`      : convergence tolerance (default: 1e-8)
  `maxiter`  : maximum iteration count (default: 100)
  `verbose`  : print debug information at each iteration (default: false)
  `debug`    : calculate and output cost function value at each iteration (default: false)
Outputs:
  `xhat`     : SBL-RWL1 estimate of x
  `Xhat`     : history of iterate for x, n x niter
  `What`     : history of iterate for w, n x niter
Reference:
  [2]        : D Wipf and S Nagarajan, 'Iterative reweighted l1 and l2 methods
               for finding sparse solutions,' IEEE J Selected Topics in Signal
               Processing, 2010.
"""
function sbl_rwl1(y, Φ, λ; τ = 1e-8, tol = 1e-8, maxiter = 100, verbose = false, debug = false)

    # initialization
    m, n = size(Φ)
    x = zeros(n)
    γ = ones(n)
    z = 1/4*ones(n) # TODO (makes weights in first iteration 0.5)
    γold = zeros(n)
    γsupp = trues(n)
    Xhat = zeros(n,maxiter)
    Γhat = zeros(n,maxiter)
    iter = 1
    converged = false
    cost = debug ? zeros(maxiter) : nothing

    # perform RWL1 iterations
    while !converged

        verbose ? print("Iteration $(iter): ") : nothing;

        # update x
        w = sqrt.(z[γsupp])
        xc = Variable(Int(sum(γsupp)))
        problem = minimize( sumsquares(y-Φ[:,γsupp]*xc) + 2*λ*sum(w.*abs(xc)) )
        solve!(problem, SCSSolver(verbose = false), verbose = false)
        xc = evaluate(xc)
        x = zeros(n)
        x[γsupp] = xc

        # update γ
        γ[γsupp] = z[γsupp].^(-1/2) .* abs.(x[γsupp])
        γsupp = abs.(γ) .>= τ

        # update z
        z[γsupp] = diag(Φ[:,γsupp]'*inv(λ*I+Φ[:,γsupp]*Diagonal(γ[γsupp])*Φ[:,γsupp]')*Φ[:,γsupp])

        # check convergence
        Xhat[:,iter] = x
        Γhat[:,iter] = γ
        converged = norm(γ-γold) < tol || iter >= maxiter
        if verbose
            @printf("%d coefficients in model; ", Int(sum(γsupp)));
            @printf("||γ-γold|| = %.3e.\n", norm(γ-γold));
        end
        if debug
          C = λ*I + Φ*Diagonal(γ)*Φ'
          cost[iter] = log(det(C)) + y'*inv(C)*y
        end
        iter += 1
        γold = γ[:]

    end

    if verbose && iter <= maxiter
        println("Converged in $(iter-1) iterations.")
    elseif verbose
        println("Reached maximum iterations.")
    end

    xhat = x
    Xhat = Xhat[:,1:iter-1]
    Γhat = Γhat[:,1:iter-1]
    cost = debug ? cost[1:iter-1] : nothing

    return xhat, Xhat, Γhat, cost

end

"""
Implements SBL using the RWL1 updates of [2]. Directly calculates weights without using γ (except for debug).
Inputs:
  `y`        : measurements, m x 1
  `Φ`        : dictionary, m x n
  `λ`        : data fidelity/sparsity tuning parameter
  `τ`        : threshold on γ used for pruning (default: 1e4)
  `tol`      : convergence tolerance (default: 1e-8)
  `maxiter`  : maximum iteration count (default: 100)
  `verbose`  : print debug information at each iteration (default: false)
  `debug`    : calculate and output cost function value at each iteration (default: false)
Outputs:
  `xhat`     : SBL-RWL1 estimate of x
  `Xhat`     : history of iterate for x, n x niter
  `What`     : history of iterate for w, n x niter
Reference:
  [2]        : D Wipf and S Nagarajan, 'Iterative reweighted l1 and l2 methods
               for finding sparse solutions,' IEEE J Selected Topics in Signal
               Processing, 2010.
"""
function sbl_rwl1_simple(y, Φ, λ; τ = 1e-8, tol = 1e-8, maxiter = 100, verbose = false, debug = false)

  # initialization
  m, n = size(Φ)
  x = zeros(n)
  w = ones(n)
  γ = ones(n)
  γold = zeros(n)
  γsupp = trues(n)
  Xhat = zeros(n,maxiter)
  What = zeros(n,maxiter)
  iter = 1
  converged = false
  cost = debug ? zeros(maxiter) : nothing

  # perform RWL1 iterations
  while !converged

      verbose ? print("Iteration $(iter): ") : nothing;

      # update x
      xc = Variable(Int(sum(γsupp)))
      problem = minimize( sumsquares(y-Φ[:,γsupp]*xc) + λ*sum(w[γsupp].*abs(xc)) )
      solve!(problem, SCSSolver(verbose = false), verbose = false)
      xc = evaluate(xc)
      x = zeros(n)
      x[γsupp] = xc

      # update w
      w[γsupp] = 2*sqrt.(diag(Φ[:,γsupp]'*inv(λ*I+Φ[:,γsupp]*Diagonal(γ[γsupp])*Φ[:,γsupp]')*Φ[:,γsupp]))
      γ = 2*abs.(x) ./ w
      γsupp = γ .>= τ

      # check convergence
      Xhat[:,iter] = x
      What[γsupp,iter] = w[γsupp]
      converged = (iter > 1 && norm(γ-γold) < tol) || iter >= maxiter
      if verbose
          @printf("%d coefficients in model; ", Int(sum(γsupp)));
          @printf("||γ-γold|| = %.3e.\n", norm(γ-γold));
      end
      if debug
        C = λ*I + Φ*Diagonal(γ)*Φ'
        cost[iter] = log(det(C)) + y'*inv(C)*y
      end
      iter += 1
      γold = γ[:]

  end

  if verbose && iter <= maxiter
      println("Converged in $(iter-1) iterations.")
  elseif verbose
      println("Reached maximum iterations.")
  end

  xhat = x
  Xhat = Xhat[:,1:iter-1]
  What = hcat(ones(n), What[:,1:iter-1])
  cost = cost[1:iter-1]

  return xhat, Xhat, What, cost

end