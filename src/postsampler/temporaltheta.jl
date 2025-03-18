using LinearAlgebra, Distributions, Random, ProgressMeter
import ..EFsCsn

struct TemporalThetaPostSampler
    y::Array{Float64,2}
    W::Array{Float64,2}
    feature::Array{Float64,3}
    nu2_beta::Float64
    a_sigma2::Float64
    b_sigma2::Float64
    a_omega2::Float64
    b_omega2::Float64
    T::Int
    K::Int
    dim::Int
    y_flatten::Array{Float64,1}
    feature_reshaped::Array{Float64,2}
    eigen_values_raw::Vector{Float64}
    eigen_vector::Matrix{Float64}

    function TemporalThetaPostSampler(
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

function lp_s(sampler::TemporalThetaPostSampler, s, u, rho, delta, omega2, eta)
    b = sqrt(2 / pi)
    gamma = 1 ./ sqrt.(1 .- b^2 * delta.^2)
    inv_Omega_s = (1 / sqrt(omega2)) * sampler.eigen_vector * Diagonal(sqrt.(eta * sampler.eigen_values_raw .+ (1-eta))) * sampler.eigen_vector'
    
    lp_s = - sampler.T * (sampler.K * log(omega2) - sum(log.(eta * sampler.eigen_values_raw .+ (1-eta)))) / 2
    tilde_s = inv_Omega_s * s[1, :]
    lp_s += sum(logpdf.(
        Normal.(
            gamma[1] * delta[1] * (u[1, :] - b * ones(sampler.K)),
            gamma[1] * sqrt(1-delta[1]^2)
        ),
        tilde_s
    ))
    
    for t in 2:sampler.T
        tilde_s = inv_Omega_s * (s[t, :] - rho * s[t-1, :])
        lp_s += sum(logpdf.(
            Normal.(
                gamma[t] * delta[t] * (u[t, :] - b * ones(sampler.K)),
                gamma[t] * sqrt(1 - delta[t]^2)
            ),
            tilde_s
        ))
    end
    return lp_s
end

function s_sampler!(s, sampler::TemporalThetaPostSampler, u, beta, sigma2, rho, delta, omega2, eta)
    b = sqrt(2 / pi)
    gamma = 1 ./ sqrt.(1 .- b^2 * delta.^2)
    Omega_s = sqrt(omega2) * sampler.eigen_vector * Diagonal(1.0 ./ sqrt.(eta * sampler.eigen_values_raw .+ (1-eta))) * sampler.eigen_vector'
    Omega = Symmetric(Omega_s * Omega_s)
    
    mean_s_pred = zeros(sampler.T, sampler.K)
    mean_s_filt = zeros(sampler.T, sampler.K)
    cov_s_pred = Vector{Symmetric{Float64, Matrix{Float64}}}(undef, sampler.T)
    cov_s_filt = Vector{Symmetric{Float64, Matrix{Float64}}}(undef, sampler.T)
    
    cov_s_pred[1] = Symmetric((1-delta[1]^2) * gamma[1]^2 * Omega)
    mean_s_pred[1, :] = gamma[1] * delta[1] * Omega_s * (u[1, :] .- b)
    cov_s_filt[1] = Symmetric(sigma2 * ((cov_s_pred[1] + sigma2 * I) \ cov_s_pred[1]))
    mean_s_filt[1, :] = mean_s_pred[1, :] + (1 / sigma2) * cov_s_filt[1] * (sampler.y[1, :] - sampler.feature[1, :, :] * beta - mean_s_pred[1, :])
    
    for t in 2:sampler.T
        cov_s_pred[t] = Symmetric(rho^2 * cov_s_filt[t - 1] + (1-delta[t]^2) * gamma[t]^2 * Omega)
        mean_s_pred[t, :] = rho * mean_s_filt[t-1, :] + gamma[t] * delta[t] * Omega_s * (u[t, :] .- b)
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

function u_sampler!(u, sampler::TemporalThetaPostSampler, s, rho, delta, omega2, eta)
    b = sqrt(2 / pi)
    gamma = 1 ./ sqrt.(1 .- b^2 * delta.^2)
    inv_Omega_s = (1 / sqrt(omega2)) * sampler.eigen_vector * Diagonal(sqrt.(eta * sampler.eigen_values_raw .+ (1-eta))) * sampler.eigen_vector'

    tilde_s = inv_Omega_s * s[1, :]
    conditional_mu_u = delta[1] / gamma[1] * tilde_s .+ b * delta[1]^2
    u[1, :] = conditional_mu_u + rand.(Truncated.(Normal(0, sqrt(1-delta[1]^2)), -conditional_mu_u, Inf))
    for t in 2:sampler.T
        tilde_s = inv_Omega_s * (s[t, :] - rho * s[t-1, :])
        conditional_mu_u = delta[t] / gamma[t] * tilde_s .+ b * delta[t]^2
        u[t, :] = conditional_mu_u + rand.(Truncated.(Normal(0, sqrt(1-delta[t]^2)), -conditional_mu_u, Inf))
    end
end

function beta_sampler(sampler::TemporalThetaPostSampler, s, sigma2)
    s_flatten = s'[:]
    cov_beta = inv(sampler.feature_reshaped' * sampler.feature_reshaped / sigma2 + I / sampler.nu2_beta)
    mu_beta = cov_beta * (sampler.feature_reshaped' * (sampler.y_flatten - s_flatten)) / sigma2
    beta = rand(MvNormal(mu_beta, Symmetric(cov_beta)))
    return beta
end

function sigma2_sampler(sampler::TemporalThetaPostSampler, s, beta)
    s_flatten = s'[:]
    an = sampler.a_sigma2 + sampler.K * sampler.T / 2
    bn = sampler.b_sigma2 + sum((sampler.y_flatten - s_flatten - sampler.feature_reshaped * beta).^2) / 2
    return rand(InverseGamma(an, bn))
end

function rho_sampler(sampler::TemporalThetaPostSampler, s, u, delta, omega2, eta)
    b = sqrt(2 / pi)
    gamma = 1 ./ sqrt.(1 .- b^2 * delta.^2)
    Omega_s = sqrt(omega2) * sampler.eigen_vector * Diagonal(1.0 ./ sqrt.(eta * sampler.eigen_values_raw .+ (1-eta))) * sampler.eigen_vector'
    inv_Omega_s = (1 / sqrt(omega2)) * sampler.eigen_vector * Diagonal(sqrt.(eta * sampler.eigen_values_raw .+ (1-eta))) * sampler.eigen_vector'

    prec_rho = 0
    mu_rho = 0
    for t in 2:sampler.T
        mu_s = gamma[t] * delta[t] * Omega_s * (u[t, :] - b * ones(sampler.K))
        s_scaled = (1 / (gamma[t] * sqrt(1-delta[t]^2))) * inv_Omega_s * s[t-1, :]
        prec_rho += s_scaled' * s_scaled
        mu_rho += (1 / (gamma[t] * sqrt(1-delta[t]^2))) * (s[t, :] - mu_s)' * inv_Omega_s * s_scaled
    end
    var_rho = 1 / prec_rho
    mu_rho /= prec_rho
    return rand(Truncated(Normal(mu_rho, sqrt(var_rho)), 0, 1))
end

function delta_sampler!(delta, sampler::TemporalThetaPostSampler, s, u, rho, omega2, eta, mu_delta, nu2_delta, delta_prop_scale)
    b = sqrt(2 / pi)
    inv_Omega_s = (1 / sqrt(omega2)) * sampler.eigen_vector * Diagonal(sqrt.(eta * sampler.eigen_values_raw .+ (1-eta))) * sampler.eigen_vector'
    dst_prior = Normal(mu_delta, sqrt(nu2_delta))

    for t in 1:sampler.T
        delta_curr = delta[t]
        lambda_curr = delta_curr / sqrt(1-delta_curr^2)
        gamma_curr = 1 / sqrt(1 - b^2 * delta_curr^2)
        
        lambda_prop = lambda_curr + sqrt(delta_prop_scale) * randn()
        delta_prop = lambda_prop / sqrt(1 + lambda_prop^2)
        gamma_prop = 1 / sqrt(1 - b^2 * delta_prop^2)
        
        if t == 1
            tilde_s = inv_Omega_s * s[1, :]
        else
            tilde_s = inv_Omega_s * (s[t, :] - rho * s[t-1, :])
        end

        lp_new = sum(logpdf.(
            Normal.(
                gamma_prop * delta_prop * (u[t, :] - b * ones(sampler.K)),
                gamma_prop * sqrt(1-delta_prop^2)
            ),
            tilde_s
        ))
        lp_old = sum(logpdf.(
            Normal.(
                gamma_curr * delta_curr * (u[t, :] - b * ones(sampler.K)),
                gamma_curr * sqrt(1-delta_curr^2)
            ),
            tilde_s
        ))
        log_accept_ratio = lp_new - lp_old
        log_accept_ratio += logpdf(dst_prior, lambda_prop) - logpdf(dst_prior, lambda_curr)
        threshold = log(rand())
        delta[t] = log_accept_ratio > threshold ? delta_prop : delta_curr
    end
end

function omega2_sampler(sampler::TemporalThetaPostSampler, s, u, rho, delta, omega2, eta, omega2_prop_scale, lp_old::Union{Float64, Nothing}=nothing)
    dst_curr = LogNormal_adjusted(omega2, omega2_prop_scale)
    omega2_prop = rand(dst_curr)
    dst_prop = LogNormal_adjusted(omega2_prop, omega2_prop_scale)
    dst_prior = InverseGamma(sampler.a_omega2, sampler.b_omega2)

    lp_new = lp_s(sampler, s, u, rho, delta, omega2_prop, eta)
    lp_old = isnothing(lp_old) ? lp_s(sampler, s, u, rho, delta, omega2, eta) : lp_old
    log_accept_ratio = lp_new - lp_old
    log_accept_ratio += logpdf(dst_prior, omega2_prop) - logpdf(dst_prior, omega2)
    log_accept_ratio -= logpdf(dst_curr, omega2_prop) - logpdf(dst_prop, omega2)
    threshold = log(rand())
    return log_accept_ratio > threshold ? (omega2_prop, lp_new) : (omega2, lp_old)
end

function eta_sampler(sampler::TemporalThetaPostSampler, s, u, rho, delta, omega2, eta, eta_prop_scale, lp_old::Union{Float64, Nothing}=nothing)
    dst_curr = Beta(eta_prop_scale * eta + 1e-4, eta_prop_scale * (1 - eta) + 1e-4)
    eta_prop = rand(dst_curr)
    dst_prop = Beta(eta_prop_scale * eta_prop + 1e-4, eta_prop_scale * (1 - eta_prop) + 1e-4)

    lp_new = lp_s(sampler, s, u, rho, delta, omega2, eta_prop)
    lp_old = isnothing(lp_old) ? lp_s(sampler, s, u, rho, delta, omega2, eta) : lp_old
    log_accept_ratio = lp_new - lp_old
    log_accept_ratio -= logpdf(dst_curr, eta_prop) - logpdf(dst_prop, eta)
    threshold = log(rand())
    return log_accept_ratio > threshold ? (eta_prop, lp_new) : (eta, lp_old)
end

function mu_and_nu2_delta_sampler(sampler::TemporalThetaPostSampler, delta)
    kappa0 = 0.01
    lambda = delta ./ sqrt.(1 .- delta.^2)
    a_n = 1 + 0.5 * sampler.T
    b_n = 0.01 + 0.5 * sum((lambda .- mean(lambda)).^2) + 0.5 * kappa0 * sampler.T / (sampler.T+kappa0) * mean(lambda)^2
    nu2_delta = rand(InverseGamma(a_n, b_n))
    mu_delta = rand(Normal(sampler.T / (sampler.T+kappa0) * mean(lambda), sqrt(nu2_delta / (sampler.T+kappa0))))
    return mu_delta, nu2_delta
end

function log_posterior(sampler::TemporalThetaPostSampler, s, u, beta, sigma2, rho, delta, omega2, eta, mu_delta, nu2_delta, lp_old::Union{Float64, Nothing}=nothing)
    s_flatten = s'[:]
    lp = isnothing(lp_old) ? lp_s(sampler, s, u, rho, delta, omega2, eta) : lp_old
    lp += sum(logpdf.(Normal.(zeros(sampler.T*sampler.K), 1), u[:]))
    lp += sum(logpdf.(Normal.(sampler.feature_reshaped * beta + s_flatten, sqrt(sigma2)), sampler.y_flatten))
    lp += sum(logpdf.(Normal(0.0, sqrt(sampler.nu2_beta)), beta))
    lp += logpdf(InverseGamma(sampler.a_sigma2, sampler.b_sigma2), sigma2)
    lp += sum(logpdf.(Normal(mu_delta, sqrt(nu2_delta)), delta ./ sqrt.(1 .- delta.^2)))
    lp += logpdf(InverseGamma(sampler.a_omega2, sampler.b_omega2), omega2)
    lp += logpdf(Normal(0, sqrt(nu2_delta/0.01)), mu_delta)
    lp += logpdf(InverseGamma(1, 0.01), nu2_delta)
    return lp
end

function sampling(
    sampler::TemporalThetaPostSampler,
    num_sample::Int;
    burn_in::Int=0,
    thinning::Int=1,
    delta_prop_scale::Float64=0.1,
    omega2_prop_scale::Float64=0.1,
    eta_prop_scale::Float64=0.1,
    s_init::Union{Matrix{Float64}, Nothing}=nothing,
    u_init::Union{Matrix{Float64}, Nothing}=nothing,
    beta_init::Union{Vector{Float64}, Nothing}=nothing,
    sigma2_init::Union{Float64, Nothing}=nothing,
    rho_init::Union{Float64, Nothing}=nothing,
    delta_init::Union{Vector{Float64}, Nothing}=nothing,
    omega2_init::Union{Float64, Nothing}=nothing,
    eta_init::Union{Float64, Nothing}=nothing,
    mu_delta_init::Union{Float64, Nothing}=nothing,
    nu2_delta_init::Union{Float64, Nothing}=nothing,
)
    eta_prop_scale = (1 - eta_prop_scale^2) / eta_prop_scale^2
    
    s = isnothing(s_init) ? zeros(sampler.T, sampler.K) : s_init
    u = isnothing(u_init) ? zeros(sampler.T, sampler.K) : u_init
    beta = isnothing(beta_init) ? zeros(sampler.dim) : beta_init
    sigma2 = isnothing(sigma2_init) ? 1.0 : sigma2_init
    rho = isnothing(rho_init) ? 0.5 : rho_init
    delta = isnothing(delta_init) ? zeros(sampler.T) : delta_init
    omega2 = isnothing(omega2_init) ? 1.0 : omega2_init
    eta = isnothing(eta_init) ? 0.5 : eta_init
    mu_delta = isnothing(mu_delta_init) ? 0.0 : mu_delta_init
    nu2_delta = isnothing(nu2_delta_init) ? 1.0 : nu2_delta_init
    lp = -Inf

    u_samples = zeros(Float64, (num_sample, sampler.T, sampler.K))
    s_samples = zeros(Float64, (num_sample, sampler.T, sampler.K))
    beta_samples = zeros(Float64, (num_sample, sampler.dim))
    sigma2_samples = zeros(Float64, num_sample)
    rho_samples = zeros(Float64, num_sample)
    delta_samples = zeros(Float64, (num_sample, sampler.T))
    omega2_samples = zeros(Float64, num_sample)
    eta_samples = zeros(Float64, num_sample)
    mu_delta_samples = zeros(Float64, num_sample)
    nu2_delta_samples = zeros(Float64, num_sample)
    lp_list = zeros(Float64, num_sample)

    @showprogress for i in 1:(burn_in + num_sample)
        for _ in 1:thinning
            u_sampler!(u, sampler, s, rho, delta, omega2, eta)
            s_sampler!(s, sampler, u, beta, sigma2, rho, delta, omega2, eta)
            beta = beta_sampler(sampler, s, sigma2)
            sigma2 = sigma2_sampler(sampler, s, beta)
            rho = rho_sampler(sampler, s, u, delta, omega2, eta)
            delta_sampler!(delta, sampler, s, u, rho, omega2, eta, mu_delta, nu2_delta, delta_prop_scale)
            omega2, lp = omega2_sampler(sampler, s, u, rho, delta, omega2, eta, omega2_prop_scale)
            eta, lp = eta_sampler(sampler, s, u, rho, delta, omega2, eta, eta_prop_scale, lp)
            mu_delta, nu2_delta = mu_and_nu2_delta_sampler(sampler, delta)
        end

        if i > burn_in
            u_samples[i-burn_in, :, :] = u
            s_samples[i-burn_in, :, :] = s
            beta_samples[i-burn_in, :] = beta
            sigma2_samples[i-burn_in] = sigma2
            rho_samples[i-burn_in] = rho
            delta_samples[i-burn_in, :] = delta
            omega2_samples[i-burn_in] = omega2
            eta_samples[i-burn_in] = eta
            mu_delta_samples[i-burn_in] = mu_delta
            nu2_delta_samples[i-burn_in] = nu2_delta
            lp_list[i-burn_in] = log_posterior(sampler, s, u, beta, sigma2, rho, delta, omega2, eta, mu_delta, nu2_delta, lp)
        end
    end

    return Dict(
        "u" => u_samples,
        "s" => s_samples,
        "beta" => beta_samples,
        "sigma2" => sigma2_samples,
        "rho" => rho_samples,
        "delta" => delta_samples,
        "omega2" => omega2_samples,
        "eta" => eta_samples,
        "mu_delta" => mu_delta_samples,
        "nu2_delta" => nu2_delta_samples,
        "lp" => lp_list
    )
end