using Pkg
Pkg.add(["DataFrames", "JLD2", "Serialization","CSV", "Dates", "Turing", "Distributions", "Plots", "StatsPlots", "Optim"])

using CSV, DataFrames, Turing, Distributions, Statistics

println("=== СТАДІЯ 1: Підготовка даних ===")
df = CSV.read("Synchronized_Data_.csv", DataFrame)
df = dropmissing(df, :combat_cases)
df.flu_cases = coalesce.(df.flu_cases, 100000.0)
df.c_mob = coalesce.(df.c_mob, 0.5)
df.k_pharm = coalesce.(df.k_pharm, 1.0)

df.E_c = df.combat_cases .* df.w_scale .* df.k_prior
df.I_real = (df.flu_cases .* df.c_mob .* df.k_pharm) ./ 1e6
df.D_drone = ((1:nrow(df)) ./ nrow(df)) .* 2.0

groups = [:tacmed, :painkiller, :antibiot, :lung, :digest, :cardio]
anchor_start = 96

coef_results = DataFrame(Group=String[], Base_Intercept=Float64[], Beta_Combat=Float64[], Beta_Flu=Float64[], Beta_Drone=Float64[], Vol_2024=Float64[], Vol_2025=Float64[])

println("=== СТАДІЯ 2: Генеративна модель (Розширена патологія) ===")

@model function train_warehouse(E_c, I_real, D_drone, y_obs, year_flags, group_name)
    sigma ~ Exponential(10.0)
    vol_2024 ~ truncated(Normal(0.30, 0.05), 0.1, 0.5)
    vol_2025 ~ truncated(Normal(0.10, 0.02), 0.0, 0.3)

    mu = zeros(length(y_obs))

    # -----------------------------------------------------------------
    # НОВА ЛОГІКА: Дрони впливають на весь травматичний цикл
    # -----------------------------------------------------------------
    if group_name == :tacmed
        beta_c ~ truncated(Normal(0.1, 0.05), 0, 0.5)
        beta_d ~ truncated(Normal(5.0, 2.0), 0, 20.0)
        mu = beta_c .* E_c .+ beta_d .* D_drone

    elseif group_name == :painkiller
        intercept ~ Normal(30, 15)
        beta_c ~ truncated(Normal(0.05, 0.02), 0, 0.5)
        beta_d ~ truncated(Normal(2.0, 1.0), 0, 10.0) # Вплив дронового травматизму на біль
        mu = max.(0.0, intercept .+ beta_c .* E_c .+ beta_d .* D_drone)

    elseif group_name == :antibiot
        intercept ~ Normal(0, 5)
        beta_c ~ truncated(Normal(0.05, 0.02), 0, 0.3)
        beta_f ~ truncated(Normal(500, 150), 0, 1500)
        beta_d ~ truncated(Normal(1.0, 0.5), 0, 5.0)  # Вплив уламкових поранень на інфекції
        mu = max.(0.0, intercept .+ beta_c .* E_c .+ beta_f .* I_real .+ beta_d .* D_drone)
    # -----------------------------------------------------------------

    elseif group_name == :lung
        intercept ~ Normal(20, 10)
        beta_f ~ truncated(Normal(200, 50), 0, 1000)
        mu = max.(0.0, intercept .+ beta_f .* I_real)
    elseif group_name == :digest
        intercept ~ Normal(40, 15)
        beta_c ~ truncated(Normal(0.01, 0.01), 0, 0.1)
        beta_f ~ truncated(Normal(200, 50), 0, 500)
        mu = max.(0.0, intercept .+ beta_c .* E_c .+ beta_f .* I_real)
    elseif group_name == :cardio
        intercept ~ Normal(30, 10)
        beta_c ~ truncated(Normal(0.005, 0.005), 0, 0.05)
        beta_f ~ truncated(Normal(10, 10), 0, 100)
        mu = max.(0.0, intercept .+ beta_c .* E_c .+ beta_f .* I_real)
    end

    for i in 1:length(y_obs)
        state_coverage = year_flags[i] == 2024 ? (1.0 - vol_2024) : (1.0 - vol_2025)
        y_obs[i] ~ Normal(state_coverage * mu[i], sigma)
    end
end

println("=== СТАДІЯ 3: MCMC Навчання та Реконструкція ===")

for gr in groups
    println(">>> Аналіз: $gr ...")
    train_idx = anchor_start:nrow(df)
    valid_mask = .!ismissing.(df[train_idx, gr])

    model = train_warehouse(df.E_c[train_idx][valid_mask], df.I_real[train_idx][valid_mask], df.D_drone[train_idx][valid_mask],
                            Float64.(df[train_idx, gr][valid_mask]), [i <= 148 ? 2024 : 2025 for i in (train_idx)[valid_mask]], gr)

    chain = sample(model, NUTS(), 500, progress=false)

    # Витяг
    inter = gr == :tacmed ? 0.0 : mean(chain[:intercept])
    b_c   = gr in [:tacmed, :painkiller, :antibiot, :digest, :cardio] ? mean(chain[:beta_c]) : 0.0
    b_f   = gr in [:antibiot, :lung, :digest, :cardio] ? mean(chain[:beta_f]) : 0.0
    b_d   = gr in [:tacmed, :painkiller, :antibiot] ? mean(chain[:beta_d]) : 0.0
    v_24, v_25 = mean(chain[:vol_2024]), mean(chain[:vol_2025])

    push!(coef_results, (string(gr), inter, b_c, b_f, b_d, v_24, v_25))

    # Реконструкція
    raw_demand = if gr == :tacmed
        b_c .* df.E_c .+ b_d .* df.D_drone
    elseif gr in [:painkiller, :antibiot]
        inter .+ b_c .* df.E_c .+ b_f .* df.I_real .+ b_d .* df.D_drone
    else
        inter .+ b_c .* df.E_c .+ b_f .* df.I_real
    end
    df[!, Symbol("true_", gr)] = max.(0.0, raw_demand)
end

display(coef_results)
CSV.write("Digital_Twin_Coefficients_V2.csv", coef_results)
CSV.write("Final_Digital_Twin_Full_Reconstruction_V2.csv", df)

=====================================================================
using CSV, DataFrames, Statistics

# 1. Завантажуємо результати
df_recon = CSV.read("Final_Digital_Twin_Full_Reconstruction_V2.csv", DataFrame)
df_coef = CSV.read("Digital_Twin_Coefficients_V2.csv", DataFrame)

# 2. Вибираємо період для аналізу (наприклад, весь 2025 рік)
# Це важливо, бо інтенсивність боїв та епідемій змінюється
df_2025 = filter(row -> 157 <= row.row_number <= 208, df_recon) # або за датою

# Рахуємо середні значення драйверів за цей період
avg_Ec = mean(df_2025.E_c)
avg_Ir = mean(df_2025.I_real)
avg_Dd = mean(df_2025.D_drone)

println("--- Середні показники драйверів (2025) ---")
println("Combat Index (Ec): ", round(avg_Ec, digits=2))
println("Flu Index (Ir): ", round(avg_Ir, digits=4))
println("Drone Index (Dd): ", round(avg_Dd, digits=2))
println("------------------------------------------\n")

# 3. Розрахунок внесків
contribution_table = DataFrame(
    Group = String[],
    Base_Part = Float64[],
    Combat_Part = Float64[],
    Flu_Part = Float64[],
    Drone_Part = Float64[],
    Total_Predicted = Float64[]
)

for r in eachrow(df_coef)
    gr = r.Group

    # Розраховуємо фізичний внесок кожного компонента в одиницях (упаковках)
    c_base = r.Base_Intercept
    c_combat = r.Beta_Combat * avg_Ec
    c_flu = r.Beta_Flu * avg_Ir
    c_drone = r.Beta_Drone * avg_Dd

    total = max(0.0, c_base + c_combat + c_flu + c_drone)

    push!(contribution_table, (gr, c_base, c_combat, c_flu, c_drone, total))
end

# 4. Вивід результату
println("Внесок кожного фактора у щотижневу потребу (в одиницях/упаковках):")
display(contribution_table)

# 5. Збереження для диплома
CSV.write("Factor_Contribution_Analysis_2025.csv", contribution_table)
=====================================================================
using CSV, DataFrames, Turing, Distributions, Statistics, Dates

println("=== СТАДІЯ 1: Підготовка даних та Feature Engineering ===")

# 1. Завантаження
df = CSV.read("Synchronized_Data_.csv", DataFrame)
df = dropmissing(df, :combat_cases)

# Запобіжники
df.flu_cases = coalesce.(df.flu_cases, 100000.0)
df.c_mob = coalesce.(df.c_mob, 0.5)
df.k_pharm = coalesce.(df.k_pharm, 1.0)

# 2. Розрахунок макро-індексів
df.E_c = df.combat_cases .* df.w_scale .* df.k_prior
df.I_real = (df.flu_cases .* df.c_mob .* df.k_pharm) ./ 1e6
df.D_drone = ((1:nrow(df)) ./ nrow(df)) .* 2.0

groups = [:tacmed, :painkiller, :antibiot, :lung, :digest, :cardio]
anchor_start = 96 # Початок 2024 року

# Каркаси для результатів
coef_results = DataFrame(Group=String[], Base_Intercept=Float64[], Beta_Combat=Float64[], Beta_Flu=Float64[], Beta_Drone=Float64[], Vol_2024=Float64[], Vol_2025=Float64[])

println("=== СТАДІЯ 2: Визначення Байєсівської моделі ===")

@model function train_warehouse(E_c, I_real, D_drone, y_obs, year_flags, group_name)
    sigma ~ Exponential(10.0)
    vol_2024 ~ truncated(Normal(0.30, 0.05), 0.1, 0.5)
    vol_2025 ~ truncated(Normal(0.10, 0.02), 0.0, 0.3)

    mu = zeros(length(y_obs))

    if group_name == :tacmed
        beta_c ~ truncated(Normal(0.1, 0.05), 0, 0.5)
        beta_d ~ truncated(Normal(5.0, 2.0), 0, 20.0)
        mu = beta_c .* E_c .+ beta_d .* D_drone

    elseif group_name == :painkiller
        intercept ~ Normal(30, 15)
        beta_c ~ truncated(Normal(0.05, 0.02), 0, 0.5)
        beta_d ~ truncated(Normal(2.0, 1.0), 0, 10.0)
        mu = max.(0.0, intercept .+ beta_c .* E_c .+ beta_d .* D_drone)

    elseif group_name == :antibiot
        intercept ~ Normal(0, 5)
        beta_c ~ truncated(Normal(0.05, 0.02), 0, 0.3)
        beta_f ~ truncated(Normal(500, 150), 0, 1500)
        beta_d ~ truncated(Normal(1.0, 0.5), 0, 5.0)
        mu = max.(0.0, intercept .+ beta_c .* E_c .+ beta_f .* I_real .+ beta_d .* D_drone)

    elseif group_name == :lung
        intercept ~ Normal(20, 10)
        beta_f ~ truncated(Normal(200, 50), 0, 1000)
        mu = max.(0.0, intercept .+ beta_f .* I_real)

    elseif group_name == :digest
        intercept ~ Normal(40, 15)
        beta_c ~ truncated(Normal(0.01, 0.01), 0, 0.1)
        beta_f ~ truncated(Normal(200, 50), 0, 500)
        mu = max.(0.0, intercept .+ beta_c .* E_c .+ beta_f .* I_real)

    elseif group_name == :cardio
        intercept ~ Normal(30, 10)
        beta_c ~ truncated(Normal(0.005, 0.005), 0, 0.05)
        beta_f ~ truncated(Normal(10, 10), 0, 100)
        mu = max.(0.0, intercept .+ beta_c .* E_c .+ beta_f .* I_real)
    end

    for i in 1:length(y_obs)
        state_coverage = year_flags[i] == 2024 ? (1.0 - vol_2024) : (1.0 - vol_2025)
        y_obs[i] ~ Normal(state_coverage * mu[i], sigma)
    end
end

println("=== СТАДІЯ 3: Навчання та Реконструкція ===")

for gr in groups
    println(">>> Обробка групи: $gr ...")

    # 1. Навчання
    train_idx = anchor_start:nrow(df)
    valid_mask = .!ismissing.(df[train_idx, gr])

    y_train = Float64.(df[train_idx, gr][valid_mask])
    Ec_train = df.E_c[train_idx][valid_mask]
    Ir_train = df.I_real[train_idx][valid_mask]
    Dd_train = df.D_drone[train_idx][valid_mask]
    year_flags = [i <= 148 ? 2024 : 2025 for i in (train_idx)[valid_mask]]

    model = train_warehouse(Ec_train, Ir_train, Dd_train, y_train, year_flags, gr)
    chain = sample(model, NUTS(), 500, progress=false)

    # 2. Екстракція коефіцієнтів (Виправлений метод)
    p_names = names(chain, :parameters)
    inter = :intercept in p_names ? mean(chain[:intercept]) : 0.0
    b_c   = :beta_c in p_names ? mean(chain[:beta_c]) : 0.0
    b_f   = :beta_f in p_names ? mean(chain[:beta_f]) : 0.0
    b_d   = :beta_d in p_names ? mean(chain[:beta_d]) : 0.0
    v_24  = mean(chain[:vol_2024])
    v_25  = mean(chain[:vol_2025])

    push!(coef_results, (string(gr), inter, b_c, b_f, b_d, v_24, v_25))

    # 3. Реконструкція для всього періоду
    raw_demand = if gr == :tacmed
        b_c .* df.E_c .+ b_d .* df.D_drone
    elseif gr in [:painkiller, :antibiot]
        inter .+ b_c .* df.E_c .+ b_f .* df.I_real .+ b_d .* df.D_drone
    elseif gr == :lung
        inter .+ b_f .* df.I_real
    else
        inter .+ b_c .* df.E_c .+ b_f .* df.I_real
    end

    df[!, Symbol("true_", gr)] = max.(0.0, raw_demand)

    # Поділ на волонтерське/державне
    vol_col = zeros(nrow(df))
    state_col = zeros(nrow(df))
    for i in 1:nrow(df)
        v_pct = i <= 52 ? 0.90 : (i <= 104 ? 0.75 : (i <= 156 ? 0.30 : 0.10))
        vol_col[i]   = df[i, Symbol("true_", gr)] * v_pct
        state_col[i] = df[i, Symbol("true_", gr)] * (1.0 - v_pct)
    end
    df[!, Symbol("vol_", gr)] = vol_col
    df[!, Symbol("state_", gr)] = state_col
end

println("=== СТАДІЯ 4: Декомпозиція внеску (Contribution Analysis) ===")

# Беремо середні значення 2025 року (рядки 157-208)
df_2025 = df[157:min(208, nrow(df)), :]
avg_Ec, avg_Ir, avg_Dd = mean(df_2025.E_c), mean(df_2025.I_real), mean(df_2025.D_drone)

contribution_results = DataFrame(Group=String[], Base=Float64[], Combat=Float64[], Flu=Float64[], Drone=Float64[], Total=Float64[])

for r in eachrow(coef_results)
    c_base = r.Base_Intercept
    c_comb = r.Beta_Combat * avg_Ec
    c_flu  = r.Beta_Flu * avg_Ir
    c_dron = r.Beta_Drone * avg_Dd
    total  = max(0.0, c_base + c_comb + c_flu + c_dron)
    push!(contribution_results, (r.Group, c_base, c_comb, c_flu, c_dron, total))
end

println("\n--- ТАБЛИЦЯ КОЕФІЦІЄНТІВ (BURN RATES) ---")
display(coef_results)

println("\n--- ДЕКОМПОЗИЦІЯ ВНЕСКУ (ОДИНИЦІ/ТИЖДЕНЬ, 2025 РІК) ---")
display(contribution_results)

# Збереження
CSV.write("Digital_Twin_Coefficients_V2.csv", coef_results)
CSV.write("Final_Digital_Twin_Full_Reconstruction_V2.csv", df)
CSV.write("Factor_Contribution_Analysis_2025.csv", contribution_results)

println("\nУспішно! Всі три файли збережено та готові до аналізу.")

