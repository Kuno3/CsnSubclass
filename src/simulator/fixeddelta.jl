using Random, Distributions, LinearAlgebra

struct FixedDeltaSimulator
    T::Int
    W::Array{Float64,2}
    K::Int
    feature::Array{Float64,3}
    dim::Int
    beta::Array{Float64,1}
    sigma2::Float64
    rhoT::Float64
    delta::Float64
    tau2::Float64
    rhoS::Float64
    theta_init::Array{Float64,1}
    theta::Array{Float64,2}
    y::Array{Float64,2}
    alpha::Array{Float64,2}

    function FixedDeltaSimulator(T, W, feature; beta=nothing, sigma2=nothing, rhoT=nothing, delta=nothing, tau2=nothing, rhoS=nothing, seed=1234, theta_init=nothing)
        Random.seed!(seed)
        K = size(W, 1)
        dim = size(feature, 3)
        beta = isnothing(beta) ? randn(dim) : beta
        sigma2 = isnothing(sigma2) ? rand(InverseGamma(1, 0.01)) : sigma2
        rhoT = isnothing(rhoT) ? rand() : rhoT
        delta = isnothing(delta) ? 2 * rand() - 1 : delta
        tau2 = isnothing(tau2) ? rand(InverseGamma(1, 0.01)) : tau2
        rhoS = isnothing(rhoS) ? rand() : rhoS
        theta_init = isnothing(theta_init) ? zeros(K) : theta_init
        new(T, W, K, feature, dim, beta, sigma2, rhoT, delta, tau2, rhoS, theta_init, zeros(T, K), zeros(T, K), zeros(T, K))
    end
end

function simulate(simulator::FixedDeltaSimulator)
    b = sqrt(2 / pi)
    Q = Symmetric(simulator.rhoS * (Diagonal(sum(simulator.W, dims=2)[:]) - simulator.W) + (1-simulator.rhoS) * I(simulator.K))
    Omega = simulator.tau2 * inv(Q)
    Omega_s = sqrt(Omega)
    gamma = 1 / sqrt(1-b^2*simulator.delta^2)

    simulator.alpha[1, :] = abs.(rand(Normal(0, 1), simulator.K))
    simulator.theta[1, :] = simulator.rhoT * simulator.theta_init + gamma * Omega_s * (sqrt(1-simulator.delta^2) * rand(Normal(0, 1), simulator.K) + simulator.delta * (simulator.alpha[1, :] .- b))
    simulator.y[1, :] = simulator.feature[1, :, :] * simulator.beta + simulator.theta[1, :] + sqrt(simulator.sigma2) * randn(simulator.K)

    for t in 2:simulator.T
        simulator.alpha[t, :] = abs.(rand(Normal(0, 1), simulator.K))
        simulator.theta[t, :] = simulator.rhoT * simulator.theta[t-1, :] + gamma * Omega_s * (sqrt(1-simulator.delta^2) * rand(Normal(0, 1), simulator.K) + simulator.delta * (simulator.alpha[t, :] .- b))
        simulator.y[t, :] = simulator.feature[t, :, :] * simulator.beta + simulator.theta[t, :] + sqrt(simulator.sigma2) * randn(simulator.K)
    end
end