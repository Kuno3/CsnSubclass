using LinearAlgebra

function logsumexp(values::Array{Float64})
    max_val = maximum(values)
    sum_exp = sum(exp.(values .- max_val))
    return log(sum_exp) + max_val
end

function LogNormal_adjusted(mean::Float64, variance::Float64)::LogNormal
    # 入力の検証
    if mean <= 0.0
        throw(ArgumentError("平均は正の値でなければなりません。"))
    end
    if variance <= 0.0
        throw(ArgumentError("分散は正の値でなければなりません。"))
    end
    
    # パラメータの計算
    sigma2 = log(1 + variance / mean^2)
    mu = log(mean) - 0.5 * sigma2
    
    # 対数正規分布の作成
    return LogNormal(mu, sqrt(sigma2))
end

function create_D(N, a)
    matrix = zeros(Float64, N, N)
        for i in 1:N
        matrix[i, i] = (i < N) ? (1 + a^2) : 1
        if i > 1
            matrix[i, i - 1] = -a
        end
        if i < N
            matrix[i, i + 1] = -a
        end
    end
    return matrix
end