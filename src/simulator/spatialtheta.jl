using Random, Distributions, LinearAlgebra

struct SpatialThetaSimulator
    T::Int
    W::Array{Float64,2}
    K::Int
    feature::Array{Float64,3}
    dim::Int
    beta::Array{Float64,1}
    sigma2::Float64
    rho::Float64
    delta::Array{Float64,1}
    omega2::Float64
    eta::Float64
    s_init::Array{Float64,1}
    s::Array{Float64,2}
    y::Array{Float64,2}
    u::Array{Float64,2}

    function SpatialThetaSimulator(T, W, feature; beta=nothing, sigma2=nothing, rho=nothing, delta=nothing, omega2=nothing, eta=nothing, seed=1234, s_init=nothing)
        Random.seed!(seed)
        K = size(W, 1)
        dim = size(feature, 3)
        beta = isnothing(beta) ? randn(dim) : beta
        sigma2 = isnothing(sigma2) ? rand(InverseGamma(1, 0.01)) : sigma2
        rho = isnothing(rho) ? rand() : rho
        delta = isnothing(delta) ? 2 * rand(K) .- 1 : delta
        omega2 = isnothing(omega2) ? rand(InverseGamma(1, 0.01)) : omega2
        eta = isnothing(eta) ? rand() : eta
        s_init = isnothing(s_init) ? zeros(K) : s_init
        new(T, W, K, feature, dim, beta, sigma2, rho, delta, omega2, eta, s_init, zeros(T, K), zeros(T, K), zeros(T, K))
    end
end

function simulate(simulator::SpatialThetaSimulator)
    b = sqrt(2 / pi)
    Q = Symmetric(simulator.eta * (Diagonal(sum(simulator.W, dims=2)[:]) - simulator.W) + (1-simulator.eta) * I(simulator.K))
    Omega = simulator.omega2 * inv(Q)
    Omega_s = sqrt(Omega)
    Delta = Diagonal(simulator.delta)
    Gamma = Diagonal(1 ./ sqrt.(1 .- b^2 .* simulator.delta.^2))
    
    simulator.u[1, :] = abs.(rand(Normal(0, 1), simulator.K))
    simulator.s[1, :] = simulator.rho * simulator.s_init + Omega_s * Gamma * (sqrt(I-Delta.^2) * rand(Normal(0, 1), simulator.K) + Delta * (simulator.u[1, :] .- b))
    simulator.y[1, :] = simulator.feature[1, :, :] * simulator.beta + simulator.s[1, :] + sqrt(simulator.sigma2) * randn(simulator.K)

    for t in 2:simulator.T
        simulator.u[t, :] = abs.(rand(Normal(0, 1), simulator.K))
        simulator.s[t, :] = simulator.rho * simulator.s[t-1, :] + Omega_s * Gamma * (sqrt(I-Delta.^2) * rand(Normal(0, 1), simulator.K) + Delta * (simulator.u[t, :] .- b))
        simulator.y[t, :] = simulator.feature[t, :, :] * simulator.beta + simulator.s[t, :] + sqrt(simulator.sigma2) * randn(simulator.K)
    end
end