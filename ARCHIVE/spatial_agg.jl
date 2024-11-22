using JuMP
using Gurobi
using DataFrames
using Statistics

function aggregate_nodes(df_solar::DataFrame, df_wind::DataFrame, N_prime::Int, distances::Matrix{Float64}, lambda1::Float64, lambda2::Float64, lambda3::Float64)
    N = size(df_solar, 2)
    T = size(df_solar, 1)
    G = ["solar", "wind"]

    # Initialize the model
    model = Model(Gurobi.Optimizer)

    # Define variables
    @variable(model, alpha[1:N, 1:N_prime], Bin)
    @variable(model, C_agg[1:N_prime, 1:length(G), 1:T])

    # Objective components
    NRMSE_av = sum(sum(alpha[n, n_prime] * (1 / length(G)) *
                       sum(
                           (sum((df_solar[t, n] - C_agg[n_prime, 1, t])^2 for t in 1:T) / T)^0.5 /
                           (maximum(df_solar[:, n]) - minimum(df_solar[:, n]))
                           for g in 1:length(G)
                       )
                       for n_prime in 1:N_prime
    ) for n in 1:N)

    correlation_error = sum(
        alpha[n, n_prime] * alpha[m, m_prime] * (1 / (length(G) * (length(G) - 1))) * sum(
            sum(
                abs(cor(df_solar[:, n], df_wind[:, m]) - cor(C_agg[n_prime, 1, :], C_agg[m_prime, 2, :]))
                for h in 1:length(G) if h != g
            )
            for g in 1:length(G)
        )
        for n in 1:N for m in 1:N if m != n for n_prime in 1:N_prime for m_prime in 1:N_prime
    )

    spatial_distance_error = sum(alpha[n, n_prime] * distances[n, n_prime] for n in 1:N for n_prime in 1:N_prime)

    # Set objective
    @objective(model, Min, lambda1 * NRMSE_av + lambda2 * correlation_error + lambda3 * spatial_distance_error)

    # Add constraints
    @constraint(model, [n in 1:N], sum(alpha[n, n_prime] for n_prime in 1:N_prime) == 1)
    @constraint(model, [n_prime in 1:N_prime, g in 1:length(G), t in 1:T],
        C_agg[n_prime, g, t] == sum(alpha[n, n_prime] * df_solar[t, n] for n in 1:N) / sum(alpha[n, n_prime] for n in 1:N))

    # Optimize the model
    optimize!(model)

    # Extract results
    alpha_result = zeros(Int, N, N_prime)
    for n in 1:N
        for n_prime in 1:N_prime
            if value(alpha[n, n_prime]) > 0.5
                alpha_result[n, n_prime] = 1
            end
        end
    end

    df_solar_agg = DataFrame(zeros(T, N_prime))
    df_wind_agg = DataFrame(zeros(T, N_prime))
    for n_prime in 1:N_prime
        for t in 1:T
            df_solar_agg[t, n_prime] = value(C_agg[n_prime, 1, t])
            df_wind_agg[t, n_prime] = value(C_agg[n_prime, 2, t])
        end
    end

    return df_solar_agg, df_wind_agg, alpha_result
end