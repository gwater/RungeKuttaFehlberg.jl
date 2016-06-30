module RungeKuttaFehlberg


export rkf45_step


"""
    calulate_steps(f, x, t, dt)

Computes and returns the 4th- and 5th-order Runge-Kutta estimates for `dx` in the differential equation `dx / dt = f(x, t)` using Fehlberg's coefficients [1].

[1] E. Fehlberg, Computing **6**, 61 (1970).
"""
function calculate_steps(f::Function, x, t::Float64, dt::Float64)
            k1 = f(x, t)
            k2 = f(x .+ k1 .* (dt * 1 / 4),
                   t + dt / 4)
            k3 = f(x .+ k1 .* (dt * 3 / 32)
                     .+ k2 .* (dt * 9 / 32),
                   t + 3 / 8 * dt)
            k4 = f(x .+ k1 .* (dt * 1932 / 2197)
                     .- k2 .* (dt * 7200 / 2197)
                     .+ k3 .* (dt * 7296 / 2197),
                   t + 12 / 13 * dt)
            k5 = f(x .+ k1 .* (dt * 439 / 216)
                     .- k2 .* (dt * 8)
                     .+ k3 .* (dt * 3680 / 513)
                     .- k4 .* (dt * 845 / 4104),
                   t + dt)
            k6 = f(x .- k1 .* (dt * 8 / 27)
                     .+ k2 .* (dt * 2)
                     .- k3 .* (dt * 3544 / 2565)
                     .+ k4 .* (dt * 1859 / 4104)
                     .- k5 .* (dt * 11 / 40),
                   t + dt / 2)
            step_rk4 = (k1 .* (dt .* 25 / 216)
                        .+ k3 .* (dt * 1408 / 2565)
                        .+ k4 .* (dt * 2197 / 4104)
                        .- k5 .* (dt / 5))
            step_rk5 = (k1 .* (dt * 16 / 135)
                        .+ k3 .* (dt * 6656 / 12825)
                        .+ k4 .* (dt * 28561 / 56430)
                        .- k5 .* (dt * 9 / 50)
                        .+ k6 .* (dt * 2 / 55))
            return step_rk4, step_rk5
end


"Default metric for `rkf45_step()`, based on the `l1`-norm (aka 'Manhattan' distance)."
function l1_metric{T}(x1::T, x2::T)
    return vecnorm(x1 .- x2, 1)
end


"""
    rkf45_step(f, x, t, tolerance, dt[, error, safety])

Computes the 5th-order Runge-Kutta estimate for `dx` in the differential equation `dx / dt = f(x, t)` with an adaptive time step method (RKF45).
Returns the estimate for `dx` and the associated time step `dt`, in that order.

# Arguments

* `f::Function`: the time derivative of `x`, must take two arguments: a state `x` and a time `t`. Should return a subtype of `typeof(x)`.
* `x`: the state variable
* `t::Float64`: the time associated with `x`
* `tolerance::Float64`: target value for the error function, `dt` will be adapted to achieve this value
* `dt::Float64`: estimate for an appropriate timestep, which should be the value for `dt` which was returned by the previous step. If `f` is well-behaved, `dt` will converge from any initial (positive) value after few steps.
* `error::Function`: a metric for the difference between two estimates of `dx`. Must take both estimates as arguments and return a positive (non-zero) number. (Default is the `l1`-norm.)
* `safety::Float64`: parameter for the timestep estimator. Must be greater than 0 and smaller than 1. Small values lead to more conservative (smaller) timestep estimates which can improve convergence. (The default, 0.9, is almost always fine.)
"""
function rkf45_step(f::Function,
                    x,
                    t::Float64,
                    tolerance::Float64,
                    dt::Float64,
                    error_norm::Function = l1_metric,
                    safety::Float64 = 0.9)
    step_rk4, step_rk5 = calculate_steps(f, x, t, dt)
    err = error_norm(step_rk4, step_rk5)
    while err > tolerance
        dt *= safety * (tolerance / err)^(1 / 5)
        step_rk4, step_rk5 = calculate_steps(f, x, t, dt)
        err = error_norm(step_rk4, step_rk5)
    end
    dt *= safety * (tolerance / err)^(1 / 4)
    if dt > 1
        dt = 1.0
    end
    return step_rk5, dt
end

end # module
