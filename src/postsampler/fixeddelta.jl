using LinearAlgebra, Distributions, Random, ProgressMeter
import ..EFsCsn

struct FixedDeltaPostSampler
    y::Array{Float64,2}
    W::Array{Float64,2}
    feature::Array{Float64,3}
    nu2_beta::Float64
    a_sigma2::Float64
    b_sigma2::Float64
    a_tau2::Float64
    b_tau2::Float64
    T::Int
    K::Int
    dim::Int
    y_flatten::Array{Float64,1}
    feature_reshaped::Array{Float64,2}
    eigen_values_raw::Vector{Float64}
    eigen_vector::Matrix{Float64}

    function FixedDeltaPostSampler(
        y,
        W,
        feature;
        nu2_beta=100.0,
        a_sigma2=1.0,
        b_sigma2=0.01,
        a_tau2=1.0,
        b_tau2=0.01,
        seed=1234
    )
        Random.seed!(seed)
        T = size(y, 1)
        K = size(W, 1)
        dim = size(feature, 3)
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
            a_tau2,
            b_tau2,
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

function lp_theta(sampler::FixedDeltaPostSampler, theta, alpha, rhoT, delta, tau2, rhoS)
    b = sqrt(2 / pi)
    gamma = 1 / sqrt(1-b^2*delta^2)
    inv_Omega_s = (1 / sqrt(tau2)) * sampler.eigen_vector * Diagonal(sqrt.(rhoS * sampler.eigen_values_raw .+ (1-rhoS))) * sampler.eigen_vector'
    
    lp_theta = - sampler.T * (sampler.K * log(tau2) - sum(log.(rhoS * sampler.eigen_values_raw .+ (1-rhoS)))) / 2
    tilde_theta = inv_Omega_s * theta[1, :]
    lp_theta += sum(logpdf.(
        Normal.(
            gamma * delta * (alpha[1, :] - b * ones(sampler.K)),
            gamma * sqrt(1-delta^2)
        ),
        tilde_theta
    ))
    
    for t in 2:sampler.T
        tilde_theta = inv_Omega_s * (theta[t, :] - rhoT * theta[t-1, :])
        lp_theta += sum(logpdf.(
            Normal.(
                gamma * delta * (alpha[t, :] - b * ones(sampler.K)),
                gamma * sqrt(1-delta^2)
            ),
            tilde_theta
        ))
    end
    return lp_theta
end

function theta_sampler!(theta, sampler::FixedDeltaPostSampler, alpha, beta, sigma2, rhoT, delta, tau2, rhoS)
    b = sqrt(2 / pi)
    gamma = 1.0 / sqrt(1-b^2*delta^2)
    Omega_s = sqrt(tau2) * sampler.eigen_vector * Diagonal(1.0 ./ sqrt.(rhoS * sampler.eigen_values_raw .+ (1-rhoS))) * sampler.eigen_vector'
    cov_theta_pred_1 = Symmetric((1-delta^2) * gamma^2 * Omega_s * Omega_s)
    
    mean_theta_pred = zeros(sampler.T, sampler.K)
    mean_theta_filt = zeros(sampler.T, sampler.K)
    cov_theta_pred = Vector{Symmetric{Float64, Matrix{Float64}}}(undef, sampler.T)
    cov_theta_filt = Vector{Symmetric{Float64, Matrix{Float64}}}(undef, sampler.T)
    
    cov_theta_pred[1] = cov_theta_pred_1
    mean_theta_pred[1, :] = gamma * delta * Omega_s * (alpha[1, :] .- b)
    cov_theta_filt[1] = Symmetric(sigma2 * ((cov_theta_pred[1] + sigma2 * I) \ cov_theta_pred[1]))
    mean_theta_filt[1, :] = mean_theta_pred[1, :] + (1 / sigma2) * cov_theta_filt[1] * (sampler.y[1, :] - sampler.feature[1, :, :] * beta - mean_theta_pred[1, :])
    
    for t in 2:sampler.T
        cov_theta_pred[t] = Symmetric(rhoT^2 * cov_theta_filt[t - 1] + cov_theta_pred_1)
        mean_theta_pred[t, :] = rhoT * mean_theta_filt[t-1, :] + gamma * delta * Omega_s * (alpha[t, :] .- b)
        cov_theta_filt[t] = Symmetric(sigma2 * ((cov_theta_pred[t] + sigma2 * I) \ cov_theta_pred[t]))
        mean_theta_filt[t, :] = mean_theta_pred[t, :] + (1 / sigma2) * cov_theta_filt[t] * (sampler.y[t, :] - sampler.feature[t, :, :] * beta - mean_theta_pred[t, :])
    end
    
    theta[sampler.T, :] = rand(MvNormal(mean_theta_filt[sampler.T, :], cov_theta_filt[sampler.T]))
    for t in sampler.T-1:-1:1
        gain = rhoT * cov_theta_filt[t] / cov_theta_pred[t+1]
        mean_theta_ffbs = mean_theta_filt[t, :] + gain * (theta[t+1, :] - mean_theta_pred[t+1, :])
        cov_theta_ffbs = Symmetric(cov_theta_filt[t] - gain * cov_theta_pred[t+1] * gain')
        theta[t, :] = rand(MvNormal(mean_theta_ffbs, cov_theta_ffbs))
    end
end

function alpha_sampler!(alpha, sampler::FixedDeltaPostSampler, theta, rhoT, delta, tau2, rhoS)
    b = sqrt(2 / pi)
    gamma = 1 / sqrt(1-b^2*delta^2)
    inv_Omega_s = (1 / sqrt(tau2)) * sampler.eigen_vector * Diagonal(sqrt.(rhoS * sampler.eigen_values_raw .+ (1-rhoS))) * sampler.eigen_vector'

    tilde_theta = inv_Omega_s * theta[1, :]
    conditional_mu_alpha = delta / gamma * tilde_theta .+ b * delta^2
    alpha[1, :] = conditional_mu_alpha + rand.(Truncated.(Normal(0, sqrt(1-delta^2)), -conditional_mu_alpha, Inf))
    for t in 2:sampler.T
        tilde_theta = inv_Omega_s * (theta[t, :] - rhoT * theta[t-1, :])
        conditional_mu_alpha = delta / gamma * tilde_theta .+ b * delta^2
        alpha[t, :] = conditional_mu_alpha + rand.(Truncated.(Normal(0, sqrt(1-delta^2)), -conditional_mu_alpha, Inf))
    end
end

function beta_sampler(sampler::FixedDeltaPostSampler, theta, sigma2)
    theta_flatten = theta'[:]
    cov_beta = inv(sampler.feature_reshaped' * sampler.feature_reshaped / sigma2 + I / sampler.nu2_beta)
    mu_beta = cov_beta * (sampler.feature_reshaped' * (sampler.y_flatten - theta_flatten)) / sigma2
    beta = rand(MvNormal(mu_beta, Symmetric(cov_beta)))
    return beta
end

function sigma2_sampler(sampler::FixedDeltaPostSampler, theta, beta)
    theta_flatten = theta'[:]
    an = sampler.a_sigma2 + sampler.K * sampler.T / 2
    bn = sampler.b_sigma2 + sum((sampler.y_flatten - theta_flatten - sampler.feature_reshaped * beta).^2) / 2
    return rand(InverseGamma(an, bn))
end

function rhoT_sampler(sampler::FixedDeltaPostSampler, theta, alpha, delta, tau2, rhoS)
    b = sqrt(2 / pi)
    gamma = 1 / sqrt(1-b^2*delta^2)
    Omega_s = sqrt(tau2) * sampler.eigen_vector * Diagonal(1.0 ./ sqrt.(rhoS * sampler.eigen_values_raw .+ (1-rhoS))) * sampler.eigen_vector'
    inv_Omega_s = (1 / sqrt(tau2)) * sampler.eigen_vector * Diagonal(sqrt.(rhoS * sampler.eigen_values_raw .+ (1-rhoS))) * sampler.eigen_vector'

    prec_rhoT = 0
    mu_rhoT = 0
    for t in 2:sampler.T
        mu_theta = gamma * delta * Omega_s * (alpha[t, :] - b * ones(sampler.K))
        theta_scaled = (1 / (gamma * sqrt(1-delta^2))) * inv_Omega_s * theta[t-1, :]
        prec_rhoT += theta_scaled' * theta_scaled
        mu_rhoT += (1 / (gamma * sqrt(1-delta^2))) * (theta[t, :] - mu_theta)' * inv_Omega_s * theta_scaled
    end
    var_rhoT = 1 / prec_rhoT
    mu_rhoT /= prec_rhoT
    return rand(Truncated(Normal(mu_rhoT, sqrt(var_rhoT)), 0, 1))
end

function delta_sampler(sampler::FixedDeltaPostSampler, theta, alpha, rhoT, delta, tau2, rhoS, mu_delta, nu2_delta, delta_prop_scale, lp_old::Union{Float64, Nothing}=nothing)
    lambda_curr = delta / sqrt(1-delta^2)
    lambda_prop = lambda_curr + sqrt(delta_prop_scale) * randn()
    delta_prop = lambda_prop / sqrt(1 + lambda_prop^2)

    dst_prior = Normal(mu_delta, sqrt(nu2_delta))

    lp_new = lp_theta(sampler, theta, alpha, rhoT, delta_prop, tau2, rhoS)
    lp_old = isnothing(lp_old) ? lp_theta(sampler, theta, alpha, rhoT, delta, tau2, rhoS) : lp_old
    log_accept_ratio = lp_new - lp_old
    log_accept_ratio += logpdf(dst_prior, lambda_prop) - logpdf(dst_prior, lambda_curr)
    threshold = log(rand())
    return log_accept_ratio > threshold ? (delta_prop, lp_new) : (delta, lp_old)
end

function tau2_sampler(sampler::FixedDeltaPostSampler, theta, alpha, rhoT, delta, tau2, rhoS, tau2_prop_scale, lp_old::Union{Float64, Nothing}=nothing)
    dst_curr = LogNormal_adjusted(tau2, tau2_prop_scale)
    tau2_prop = rand(dst_curr)
    dst_prop = LogNormal_adjusted(tau2_prop, tau2_prop_scale)
    dst_prior = InverseGamma(sampler.a_tau2, sampler.b_tau2)

    lp_new = lp_theta(sampler, theta, alpha, rhoT, delta, tau2_prop, rhoS)
    lp_old = isnothing(lp_old) ? lp_theta(sampler, theta, alpha, rhoT, delta, tau2, rhoS) : lp_old
    log_accept_ratio = lp_new - lp_old
    log_accept_ratio += logpdf(dst_prior, tau2_prop) - logpdf(dst_prior, tau2)
    log_accept_ratio -= logpdf(dst_curr, tau2_prop) - logpdf(dst_prop, tau2)
    threshold = log(rand())
    return log_accept_ratio > threshold ? (tau2_prop, lp_new) : (tau2, lp_old)
end

function rhoS_sampler(sampler::FixedDeltaPostSampler, theta, alpha, rhoT, delta, tau2, rhoS, rhoS_prop_scale, lp_old::Union{Float64, Nothing}=nothing)
    dst_curr = Beta(rhoS_prop_scale * rhoS + 1e-4, rhoS_prop_scale * (1 - rhoS) + 1e-4)
    rhoS_prop = rand(dst_curr)
    dst_prop = Beta(rhoS_prop_scale * rhoS_prop + 1e-4, rhoS_prop_scale * (1 - rhoS_prop) + 1e-4)

    lp_new = lp_theta(sampler, theta, alpha, rhoT, delta, tau2, rhoS_prop)
    lp_old = isnothing(lp_old) ? lp_theta(sampler, theta, alpha, rhoT, delta, tau2, rhoS) : lp_old
    log_accept_ratio = lp_new - lp_old
    log_accept_ratio -= logpdf(dst_curr, rhoS_prop) - logpdf(dst_prop, rhoS)
    threshold = log(rand())
    return log_accept_ratio > threshold ? (rhoS_prop, lp_new) : (rhoS, lp_old)
end

function mu_and_nu2_delta_sampler(sampler::FixedDeltaPostSampler, delta)
    kappa0 = 0.01
    lambda = delta / sqrt(1 - delta^2)
    a_n = 1 + 0.5
    b_n = 0.01 + 0.5 * kappa0 / (1+kappa0) * lambda^2
    nu2_delta = rand(InverseGamma(a_n, b_n))
    mu_delta = rand(Normal(1 / (1+kappa0) * lambda, sqrt(nu2_delta / (1+kappa0))))
    return mu_delta, nu2_delta
end

function log_posterior(sampler::FixedDeltaPostSampler, theta, alpha, beta, sigma2, rhoT, delta, tau2, rhoS, mu_delta, nu2_delta, lp_old::Union{Float64, Nothing}=nothing)
    theta_flatten = theta'[:]
    lp = isnothing(lp_old) ? lp_theta(sampler, theta, alpha, rhoT, delta, tau2, rhoS) : lp_old
    lp += sum(logpdf.(Normal.(zeros(sampler.T*sampler.K), 1), alpha[:]))
    lp += sum(logpdf.(Normal.(sampler.feature_reshaped * beta + theta_flatten, sqrt(sigma2)), sampler.y_flatten))
    lp += sum(logpdf.(Normal(0.0, sqrt(sampler.nu2_beta)), beta))
    lp += logpdf(InverseGamma(sampler.a_sigma2, sampler.b_sigma2), sigma2)
    lp += logpdf(Normal(mu_delta, sqrt(nu2_delta)), delta / sqrt(1-delta^2))
    lp += logpdf(InverseGamma(sampler.a_tau2, sampler.b_tau2), tau2)
    lp += logpdf(Normal(0, sqrt(nu2_delta/0.01)), mu_delta)
    lp += logpdf(InverseGamma(1, 0.01), nu2_delta)
    return lp
end

function sampling(
    sampler::FixedDeltaPostSampler,
    num_sample::Int;
    burn_in::Int=0,
    thinning::Int=1,
    delta_prop_scale::Float64=0.1,
    tau2_prop_scale::Float64=0.1,
    rhoS_prop_scale::Float64=0.1,
    theta_init::Union{Matrix{Float64}, Nothing}=nothing,
    alpha_init::Union{Matrix{Float64}, Nothing}=nothing,
    beta_init::Union{Vector{Float64}, Nothing}=nothing,
    sigma2_init::Union{Float64, Nothing}=nothing,
    rhoT_init::Union{Float64, Nothing}=nothing,
    delta_init::Union{Float64, Nothing}=nothing,
    tau2_init::Union{Float64, Nothing}=nothing,
    rhoS_init::Union{Float64, Nothing}=nothing,
    mu_delta_init::Union{Float64, Nothing}=nothing,
    nu2_delta_init::Union{Float64, Nothing}=nothing
)
    rhoS_prop_scale = (1 - rhoS_prop_scale^2) / rhoS_prop_scale^2
    
    theta = isnothing(theta_init) ? zeros(sampler.T, sampler.K) : theta_init
    alpha = isnothing(alpha_init) ? zeros(sampler.T, sampler.K) : alpha_init
    beta = isnothing(beta_init) ? zeros(sampler.dim) : beta_init
    sigma2 = isnothing(sigma2_init) ? 1.0 : sigma2_init
    rhoT = isnothing(rhoT_init) ? 0.5 : rhoT_init
    delta = isnothing(delta_init) ? 0.0 : delta_init
    tau2 = isnothing(tau2_init) ? 1.0 : tau2_init
    rhoS = isnothing(rhoS_init) ? 0.5 : rhoS_init
    mu_delta = isnothing(mu_delta_init) ? 0.0 : mu_delta_init
    nu2_delta = isnothing(nu2_delta_init) ? 1.0 : nu2_delta_init
    lp = -Inf

    alpha_samples = zeros(Float64, (num_sample, sampler.T, sampler.K))
    theta_samples = zeros(Float64, (num_sample, sampler.T, sampler.K))
    beta_samples = zeros(Float64, (num_sample, sampler.dim))
    sigma2_samples = zeros(Float64, num_sample)
    rhoT_samples = zeros(Float64, num_sample)
    delta_samples = zeros(Float64, num_sample)
    tau2_samples = zeros(Float64, num_sample)
    rhoS_samples = zeros(Float64, num_sample)
    mu_delta_samples = zeros(Float64, num_sample)
    nu2_delta_samples = zeros(Float64, num_sample)
    lp_list = zeros(Float64, num_sample)

    @showprogress for i in 1:(burn_in + num_sample)
        for _ in 1:thinning
            alpha_sampler!(alpha, sampler, theta, rhoT, delta, tau2, rhoS)
            theta_sampler!(theta, sampler, alpha, beta, sigma2, rhoT, delta, tau2, rhoS)
            beta = beta_sampler(sampler, theta, sigma2)
            sigma2 = sigma2_sampler(sampler, theta, beta)
            rhoT = rhoT_sampler(sampler, theta, alpha, delta, tau2, rhoS)
            delta, lp = delta_sampler(sampler, theta, alpha, rhoT, delta, tau2, rhoS, mu_delta, nu2_delta, delta_prop_scale)
            tau2, lp = tau2_sampler(sampler, theta, alpha, rhoT, delta, tau2, rhoS, tau2_prop_scale, lp)
            rhoS, lp = rhoS_sampler(sampler, theta, alpha, rhoT, delta, tau2, rhoS, rhoS_prop_scale, lp)
            mu_delta, nu2_delta = mu_and_nu2_delta_sampler(sampler, delta)
        end

        if i > burn_in
            alpha_samples[i-burn_in, :, :] = alpha
            theta_samples[i-burn_in, :, :] = theta
            beta_samples[i-burn_in, :] = beta
            sigma2_samples[i-burn_in] = sigma2
            rhoT_samples[i-burn_in] = rhoT
            delta_samples[i-burn_in] = delta
            tau2_samples[i-burn_in] = tau2
            rhoS_samples[i-burn_in] = rhoS
            mu_delta_samples[i-burn_in] = mu_delta
            nu2_delta_samples[i-burn_in] = nu2_delta
            lp_list[i-burn_in] = log_posterior(sampler, theta, alpha, beta, sigma2, rhoT, delta, tau2, rhoS, mu_delta, nu2_delta, lp)
        end
    end

    return Dict(
        "alpha" => alpha_samples,
        "theta" => theta_samples,
        "beta" => beta_samples,
        "sigma2" => sigma2_samples,
        "rhoT" => rhoT_samples,
        "delta" => delta_samples,
        "tau2" => tau2_samples,
        "rhoS" => rhoS_samples,
        "mu_delta" => mu_delta_samples,
        "nu2_delta" => nu2_delta_samples,
        "lp" => lp_list
    )
end