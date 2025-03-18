using Random, Distributions, LinearAlgebra

struct ZeroThetaSimulator
    T::Int
    W::Array{Float64,2}
    K::Int
    feature::Array{Float64,3}
    dim::Int
    beta::Array{Float64,1}
    sigma2::Float64
    rho::Float64
    omega2::Float64
    eta::Float64
    s_init::Array{Float64,1}
    s::Array{Float64,2}
    y::Array{Float64,2}

    function ZeroThetaSimulator(T, W, feature; beta=nothing, sigma2=nothing, rho=nothing, omega2=nothing, eta=nothing, seed=1234, s_init=nothing)
        Random.seed!(seed)
        K = size(W, 1)
        dim = size(feature, 3)
        beta = isnothing(beta) ? randn(dim) : beta
        sigma2 = isnothing(sigma2) ? rand(InverseGamma(1, 0.01)) : sigma2
        rho = isnothing(rho) ? randn() : rho
        omega2 = isnothing(omega2) ? rand(InverseGamma(1, 0.01)) : omega2
        eta = isnothing(eta) ? rand() : eta
        s_init = isnothing(s_init) ? zeros(K) : s_init

        new(T, W, K, feature, dim, beta, sigma2, rho, omega2, eta, s_init, zeros(T, K), zeros(T, K))
    end
end

function simulate(simulator::ZeroThetaSimulator)
    Q = Symmetric(simulator.eta * (Diagonal(sum(simulator.W, dims=2)[:]) - simulator.W) + (1-simulator.eta) * I(simulator.K))
    Omega = simulator.omega2 * inv(Q)

    simulator.s[1,:] = rand(MvNormal(simulator.rho * simulator.s_init, Omega))
    simulator.y[1,:] = simulator.feature[1,:,:] * simulator.beta + simulator.s[1,:] + sqrt(simulator.sigma2) * randn(simulator.K)

    for t in 2:simulator.T
        simulator.s[t,:] = rand(MvNormal(simulator.rho * simulator.s[t-1, :], Omega))
        simulator.y[t,:] = simulator.feature[t,:,:] * simulator.beta + simulator.s[t,:] + sqrt(simulator.sigma2) * randn(simulator.K)
    end
end