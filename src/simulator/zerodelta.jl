using Random, Distributions, LinearAlgebra

struct ZeroDeltaSimulator
    T::Int
    W::Array{Float64,2}
    K::Int
    feature::Array{Float64,3}
    dim::Int
    beta::Array{Float64,1}
    sigma2::Float64
    rhoT::Float64
    tau2::Float64
    rhoS::Float64
    theta_init::Array{Float64,1}
    theta::Array{Float64,2}
    y::Array{Float64,2}

    function ZeroDeltaSimulator(T, W, feature; beta=nothing, sigma2=nothing, rhoT=nothing, tau2=nothing, rhoS=nothing, seed=1234, theta_init=nothing)
        Random.seed!(seed)
        K = size(W, 1)
        dim = size(feature, 3)
        beta = isnothing(beta) ? randn(dim) : beta
        sigma2 = isnothing(sigma2) ? rand(InverseGamma(1, 0.01)) : sigma2
        rhoT = isnothing(rhoT) ? randn() : rhoT
        tau2 = isnothing(tau2) ? rand(InverseGamma(1, 0.01)) : tau2
        rhoS = isnothing(rhoS) ? rand() : rhoS
        theta_init = isnothing(theta_init) ? zeros(K) : theta_init

        new(T, W, K, feature, dim, beta, sigma2, rhoT, tau2, rhoS, theta_init, zeros(T, K), zeros(T, K))
    end
end

function simulate(simulator::ZeroDeltaSimulator)
    Q = Symmetric(simulator.rhoS * (Diagonal(sum(simulator.W, dims=2)[:]) - simulator.W) + (1-simulator.rhoS) * I(simulator.K))
    Omega = simulator.tau2 * inv(Q)

    simulator.theta[1,:] = rand(MvNormal(simulator.rhoT * simulator.theta_init, Omega))
    simulator.y[1,:] = simulator.feature[1,:,:] * simulator.beta + simulator.theta[1,:] + sqrt(simulator.sigma2) * randn(simulator.K)

    for t in 2:simulator.T
        simulator.theta[t,:] = rand(MvNormal(simulator.rhoT * simulator.theta[t-1, :], Omega))
        simulator.y[t,:] = simulator.feature[t,:,:] * simulator.beta + simulator.theta[t,:] + sqrt(simulator.sigma2) * randn(simulator.K)
    end
end