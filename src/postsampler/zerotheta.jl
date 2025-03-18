using LinearAlgebra, Random, Distributions, ProgressMeter

struct ZeroThetaPostSampler
    y::Array{Float64, 2}
    W::Array{Float64, 2}
    feature::Array{Float64, 3}
    nu2_beta::Float64
    a_sigma2::Float64
    b_sigma2::Float64
    a_omega2::Float64
    b_omega2::Float64
    T::Int
    K::Int
    dim::Int
    y_flatten::Array{Float64, 1}
    feature_reshaped::Array{Float64, 2}
    eigen_values_raw::Vector{Float64}
    eigen_vector::Matrix{Float64}

    function ZeroThetaPostSampler(
        y,
        W,
        feature;
        nu2_beta=100.0,
        a_sigma2=1.0,
        b_sigma2=0.01,
        a_omega2=1.0,
        b_omega2=0.01,
        seed=1234
    )
        Random.seed!(seed)
        T = size(y, 1)
        K = size(W, 1)
        dim = size(feature, 3)
        y_flatten = vec(y)
        y_flatten = y'[:]
        feature_reshaped = reshape(permutedims(feature, [2, 1, 3]), T * K, dim)
        eigen_decomp = eigen(Diagonal(sum(W, dims=2)[:]) - W)
        eigen_values_raw = eigen_decomp.values
        eigen_vector = eigen_decomp.vectors

        new(
            y,
            W,
            feature,
            nu2_beta,
            a_sigma2,
            b_sigma2,
            a_omega2,
            b_omega2,
            T,
            K,
            dim,
            y_flatten,
            feature_reshaped,
            eigen_values_raw,
            eigen_vector
        )
    end
end

function lp_s(sampler::ZeroThetaPostSampler, s, rho, omega2, eta)
    inv_Omega_s = (1 / sqrt(omega2)) * sampler.eigen_vector * Diagonal(sqrt.(eta * sampler.eigen_values_raw .+ (1-eta))) * sampler.eigen_vector'
    
    lp_s = - sampler.T * (sampler.K * log(omega2) - sum(log.(eta * sampler.eigen_values_raw .+ (1-eta)))) / 2
    tilde_s = inv_Omega_s * s[1, :]
    lp_s += sum(logpdf.(
        Normal.(
            0,
            1
        ),
        tilde_s
    ))
    
    for t in 2:sampler.T
        tilde_s = inv_Omega_s * (s[t, :] - rho * s[t-1, :])
        lp_s += sum(logpdf.(
            Normal.(
                0,
                1
            ),
            tilde_s
        ))
    end
    return lp_s
end

function s_sampler!(s, sampler::ZeroThetaPostSampler, beta, sigma2, rho, omega2, eta)
    Omega = Symmetric(omega2 * sampler.eigen_vector * Diagonal(1 ./ (eta * sampler.eigen_values_raw .+ (1-eta))) * sampler.eigen_vector')
    
    mean_s_pred = zeros(sampler.T, sampler.K)
    mean_s_filt = zeros(sampler.T, sampler.K)
    cov_s_pred = Vector{Symmetric{Float64, Matrix{Float64}}}(undef, sampler.T)
    cov_s_filt = Vector{Symmetric{Float64, Matrix{Float64}}}(undef, sampler.T)
    
    cov_s_pred[1] = Omega
    mean_s_pred[1, :] = zeros(sampler.K)
    cov_s_filt[1] = Symmetric(sigma2 * ((cov_s_pred[1] + sigma2 * I) \ cov_s_pred[1]))
    mean_s_filt[1, :] = mean_s_pred[1, :] + (1 / sigma2) * cov_s_filt[1] * (sampler.y[1, :] - sampler.feature[1, :, :] * beta - mean_s_pred[1, :])
    
    for t in 2:sampler.T
        cov_s_pred[t] = Symmetric(rho^2 * cov_s_filt[t-1] + Omega)
        mean_s_pred[t, :] = rho * mean_s_filt[t-1, :]
        cov_s_filt[t] = Symmetric(sigma2 * ((cov_s_pred[t] + sigma2 * I) \ cov_s_pred[t]))
        mean_s_filt[t, :] = mean_s_pred[t, :] + (1 / sigma2) * cov_s_filt[t] * (sampler.y[t, :] - sampler.feature[t, :, :] * beta - mean_s_pred[t, :])
    end
    
    s[sampler.T, :] = rand(MvNormal(mean_s_filt[sampler.T, :], cov_s_filt[sampler.T]))
    for t in sampler.T-1:-1:1
        gain = rho * cov_s_filt[t] / cov_s_pred[t+1]
        mean_s_ffbs = mean_s_filt[t, :] + gain * (s[t+1, :] - mean_s_pred[t+1, :])
        cov_s_ffbs = Symmetric(cov_s_filt[t] - gain * cov_s_pred[t+1] * gain')
        s[t, :] = rand(MvNormal(mean_s_ffbs, cov_s_ffbs))
    end
end

function beta_sampler(sampler::ZeroThetaPostSampler, s, sigma2)
    s_flatten = s'[:]
    cov_beta = inv(sampler.feature_reshaped' * sampler.feature_reshaped / sigma2 + I / sampler.nu2_beta)
    mu_beta = cov_beta * (sampler.feature_reshaped' * (sampler.y_flatten - s_flatten)) / sigma2
    beta = rand(MvNormal(mu_beta, Symmetric(cov_beta)))
    return beta
end

function sigma2_sampler(sampler::ZeroThetaPostSampler, s, beta)
    s_flatten = s'[:]
    an = sampler.a_sigma2 + sampler.K * sampler.T / 2
    bn = sampler.b_sigma2 + sum((sampler.y_flatten - s_flatten - sampler.feature_reshaped * beta).^2) / 2
    return rand(InverseGamma(an, bn))
end

function omega2_sampler(sampler::ZeroThetaPostSampler, s, rho, eta)
    s_flatten = s'[:]
    Q = Symmetric(eta * (Diagonal(sum(sampler.W, dims=2)[:]) - sampler.W) + (1-eta) * I(sampler.K))
    D = create_D(sampler.T, rho)
    an = sampler.a_omega2 + sampler.K * sampler.T / 2
    bn = sampler.b_omega2 + (s_flatten' * (kron(D, Q) * s_flatten)) / 2
    return rand(InverseGamma(an, bn))
end

function eta_sampler(sampler::ZeroThetaPostSampler, s, rho, omega2, eta, eta_prop_scale, lp_old::Union{Float64, Nothing}=nothing)
    dst_curr = Beta(eta_prop_scale * eta + 1e-4, eta_prop_scale * (1 - eta) + 1e-4)
    eta_prop = rand(dst_curr)
    dst_prop = Beta(eta_prop_scale * eta_prop + 1e-4, eta_prop_scale * (1 - eta_prop) + 1e-4)

    lp_new = lp_s(sampler, s, rho, omega2, eta_prop)
    lp_old = isnothing(lp_old) ? lp_s(sampler, s, rho, omega2, eta) : lp_old
    log_accept_ratio = lp_new - lp_old
    log_accept_ratio -= logpdf(dst_curr, eta_prop) - logpdf(dst_prop, eta)
    threshold = log(rand())
    return log_accept_ratio > threshold ? (eta_prop, lp_new) : (eta, lp_old)
end

function rho_sampler(sampler::ZeroThetaPostSampler, s, omega2, eta)
    inv_Omega_s = (1 / sqrt(omega2)) * sampler.eigen_vector * Diagonal(sqrt.(eta * sampler.eigen_values_raw .+ (1-eta))) * sampler.eigen_vector'

    prec_rho = 0
    mu_rho = 0
    for t in 2:sampler.T
        s_scaled = inv_Omega_s * s[t-1, :]
        prec_rho += s_scaled' * s_scaled
        mu_rho += s[t, :]' * inv_Omega_s * s_scaled
    end
    var_rho = 1 / prec_rho
    mu_rho /= prec_rho
    return rand(Truncated(Normal(mu_rho, sqrt(var_rho)), 0, 1))
end

function log_posterior(sampler::ZeroThetaPostSampler, s, beta, sigma2, rho, omega2, eta, lp_old::Union{Float64, Nothing}=nothing)
    s_flatten = s'[:]
    lp = isnothing(lp_old) ? lp_s(sampler, s, rho, omega2, eta) : lp_old
    lp += sum(logpdf.(Normal.(sampler.feature_reshaped * beta + s_flatten, sqrt(sigma2)), sampler.y_flatten))
    lp += sum(logpdf.(Normal(0.0, sqrt(sampler.nu2_beta)), beta))
    lp += logpdf(InverseGamma(sampler.a_sigma2, sampler.b_sigma2), sigma2)
    lp += logpdf(InverseGamma(sampler.a_omega2, sampler.b_omega2), omega2)
    return lp
end

function sampling(
    sampler::ZeroThetaPostSampler,
    num_sample;
    burn_in=0,
    thinning=1,
    eta_prop_scale=0.1,
    s_init::Union{Matrix{Float64}, Nothing}=nothing,
    beta_init::Union{Vector{Float64}, Nothing}=nothing,
    sigma2_init::Union{Float64, Nothing}=nothing,
    rho_init::Union{Float64, Nothing}=nothing,
    omega2_init::Union{Float64, Nothing}=nothing,
    eta_init::Union{Float64, Nothing}=nothing,
)
    eta_prop_scale = (1 - eta_prop_scale^2) / eta_prop_scale^2
    
    # Initialize variables
    s = isnothing(s_init) ? zeros(sampler.T, sampler.K) : s_init
    beta = isnothing(beta_init) ? zeros(sampler.dim) : beta_init
    sigma2 = isnothing(sigma2_init) ? 1.0 : sigma2_init
    rho = isnothing(rho_init) ? 0.5 : rho_init
    omega2 = isnothing(omega2_init) ? 1.0 : omega2_init
    eta = isnothing(eta_init) ? 0.5 : eta_init
    lp = -Inf

    # Arrays to store samples
    s_samples = zeros(Float64, (num_sample, sampler.T, sampler.K))
    beta_samples = zeros(Float64, (num_sample, sampler.dim))
    sigma2_samples = zeros(Float64, num_sample)
    rho_samples = zeros(Float64, num_sample)
    omega2_samples = zeros(Float64, num_sample)
    eta_samples = zeros(Float64, num_sample)
    lp_list = zeros(Float64, num_sample)

    @showprogress for i in 1:(burn_in + num_sample)
        for _ in 1:thinning
            s_sampler!(s, sampler, beta, sigma2, rho, omega2, eta)
            beta = beta_sampler(sampler, s, sigma2)
            sigma2 = sigma2_sampler(sampler, s, beta)
            omega2 = omega2_sampler(sampler, s, rho, eta)
            rho = rho_sampler(sampler, s, omega2, eta)
            eta, lp = eta_sampler(sampler, s, rho, omega2, eta, eta_prop_scale)
        end

        if i > burn_in
            s_samples[i-burn_in, :, :] = s
            beta_samples[i-burn_in, :] = beta
            sigma2_samples[i-burn_in] = sigma2
            rho_samples[i-burn_in] = rho
            omega2_samples[i-burn_in] = omega2
            eta_samples[i-burn_in] = eta
            lp_list[i-burn_in] = log_posterior(sampler, s, beta, sigma2, rho, omega2, eta, lp)
        end
    end

    return Dict(
        "s" => s_samples,
        "beta" => beta_samples,
        "sigma2" => sigma2_samples,
        "rho" => rho_samples,
        "omega2" => omega2_samples,
        "eta" => eta_samples,
        "lp" => lp_list
    )
end