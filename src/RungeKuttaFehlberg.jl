__precompile__()

module RungeKuttaFehlberg


export rkf45_step, rkf45_step!, RKFBuffer


"""
    calulate_steps(f, x, t, dt)

Computes and returns the 4th- and 5th-order Runge-Kutta estimates for `dx` in the differential equation `dx / dt = f(x, t)` using Fehlberg's coefficients [1].

[1] E. Fehlberg, Computing **6**, 61 (1970).
"""
function calculate_steps(f::Function, x, t::Float64, dt::Float64)
            k1 = f(x, t)
            k2 = f(x .+ k1 * (dt * 1 / 4),
                   t + dt / 4)
            k3 = f(x .+ k1 * (dt * 3 / 32)
                     .+ k2 * (dt * 9 / 32),
                   t + 3 / 8 * dt)
            k4 = f(x .+ k1 * (dt * 1932 / 2197)
                     .- k2 * (dt * 7200 / 2197)
                     .+ k3 * (dt * 7296 / 2197),
                   t + 12 / 13 * dt)
            k5 = f(x .+ k1 * (dt * 439 / 216)
                     .- k2 * (dt * 8)
                     .+ k3 * (dt * 3680 / 513)
                     .- k4 * (dt * 845 / 4104),
                   t + dt)
            k6 = f(x .- k1 * (dt * 8 / 27)
                     .+ k2 * (dt * 2)
                     .- k3 * (dt * 3544 / 2565)
                     .+ k4 * (dt * 1859 / 4104)
                     .- k5 * (dt * 11 / 40),
                   t + dt / 2)
            step_rk4 = (k1 * (dt * 25 / 216)
                        .+ k3 * (dt * 1408 / 2565)
                        .+ k4 * (dt * 2197 / 4104)
                        .- k5 * (dt / 5))
            step_rk5 = (k1 * (dt * 16 / 135)
                        .+ k3 * (dt * 6656 / 12825)
                        .+ k4 * (dt * 28561 / 56430)
                        .- k5 * (dt * 9 / 50)
                        .+ k6 * (dt * 2 / 55))
            return step_rk4, step_rk5
end

immutable RKFBuffer{T}
    k1::T
    k2::T
    k3::T
    k4::T
    k5::T
    k6::T
    x_temp::T
end
RKFBuffer{T}(x::T) = RKFBuffer(similar(x), similar(x), similar(x), similar(x),
                               similar(x), similar(x), similar(x))

"""
    calulate_steps(f!, x, t, dt, b, dx4, dx5)

Like calculate_steps, but takes a mutating function `f!(x, t, dx)` which writes
the differential `dx` into its final argument.

`x` must be of a mutable type (it will not be mutated) and there must be a
`map!()` method for it.

`b` is a pre-allocated `RKFBuffer` object (it will be mutated).
The 4th and 5th order results will be written to `dx4` and  `dx5`, respectively.
"""
function calculate_steps!{T}(f!::Function, x::T, t::Float64, dt::Float64,
                             b::RKFBuffer{T}, dx4::T, dx5::T)
            f!(x, t, b.k1)
            map!((x, k1) -> x + k1 * 0.35dt, b.x_temp, x, b.k1)
            f!(b.x_temp, t + 0.25dt, b.k2)
            map!(b.x_temp, x, b.k1, b.k2) do x, k1, k2
                x + k1 * (dt * 3 / 32) + k2 * (dt * 9 / 32)
            end
            f!(b.x_temp, t + 0.375dt, b.k3)
            map!(b.x_temp, x, b.k1, b.k2, b.k3) do x, k1, k2, k3
                x + k1 * (dt * 1932 / 2197) - k2 * (dt * 7200 / 2197) +
                    k3 * (dt * 7296 / 2197)
            end
            f!(b.x_temp, t + 12 / 13 * dt, b.k4)
            map!(b.x_temp, x, b.k1, b.k2, b.k3, b.k4) do x, k1, k2, k3, k4
                x + k1 * (dt * 439 / 216) - k2 * (dt * 8) +
                    k3 * (dt * 3680 / 513) - k4 * (dt * 845 / 4104)
            end
            f!(b.x_temp, t + dt, b.k5)
            map!(b.x_temp, x, b.k1, b.k2, b.k3, b.k4, b.k5) do x, k1, k2, k3,
                                                               k4, k5
                x - k1 * (dt * 8 / 27) + k2 * (dt * 2) -
                    k3 * (dt * 3544 / 2565) + k4 * (dt * 1859 / 4104) -
                    k5 * (dt * 11 / 40)
            end
            f!(b.x_temp, t + 0.5dt, b.k6)
            map!(dx4, b.k1, b.k3, b.k4, b.k5) do k1, k3, k4, k5
                k1 * (dt * 25 / 216) + k3 * (dt * 1408 / 2565) +
                    k4 * (dt * 2197 / 4104) - k5 * 0.2dt
            end
            map!(dx5, b.k1, b.k3, b.k4, b.k5, b.k6) do k1, k3, k4, k5, k6
                k1 * (dt * 16 / 135) + k3 * (dt * 6656 / 12825) +
                    k4 * (dt * 28561 / 56430) - k5 * (dt * 9 / 50) +
                    k6 * (dt * 2 / 55)
            end
            return dx4, dx5
end

"""
Default metric for `rkf45_step()`, based on the `l1`-norm (aka 'Manhattan'
distance).
"""
function l1_metric{T}(x1::T, x2::T)
    return vecnorm(x1 .- x2, 1)
end


"""
    rkf45_step(f, x, t, tolerance, dt[, error, safety])

Computes the 5th-order Runge-Kutta estimate for `dx` in the differential equation `dx / dt = f(x, t)` with an adaptive time step method (RKF45).
Returns the estimate for `dx`, the associated time step `dt` and a suggestion for the next timestep `next_dt`, in that order.

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
                    max_dt::Float64 = 1.0,
                    safety::Float64 = 0.9)
    step_rk4, step_rk5 = calculate_steps(f, x, t, dt)
    err = error_norm(step_rk4, step_rk5)
    while err > tolerance
        dt *= safety * (tolerance / err)^(1 / 5)
        step_rk4, step_rk5 = calculate_steps(f, x, t, dt)
        err = error_norm(step_rk4, step_rk5)
    end
    next_dt = dt * safety * (tolerance / err)^(1 / 4)
    # Note that we need to catch cases where `err` approaches zero.
    return step_rk5, dt, min(next_dt, max_dt)
end

"""
    rkf45_step!(f!, x, t, tol, dt, dx4, dx5, buffer[, err_norm, max_dt, safety])

Like `rkf45_step`, but instead of dynamical allocations uses `buffer`, `dx4` and
`dx5` to store data.
"""
function rkf45_step!{T}(f!::Function,
                        x::T,
                        t::Float64,
                        tol::Float64,
                        dt::Float64,
                        dx4::T,
                        dx5::T,
                        buffer::RKFBuffer{T},
                        err_norm::Function = l1_metric,
                        max_dt::Float64 = 1.0,
                        safety::Float64 = 0.9)
    calculate_steps!(f!, x, t, dt, buffer, dx4, dx5)
    err = err_norm(dx4, dx5)
    while err > tol
        dt *= safety * (tol / err)^(1 / 5)
        calculate_steps!(f!, x, t, dt, buffer, dx4, dx5)
        err = err_norm(dx4, dx5)
    end
    next_dt = dt * safety * (tol / err)^(1 / 4)
    # Note that we need to catch cases where `err` approaches zero.
    return dx5, dt, min(next_dt, max_dt)
end

end # module
