using RungeKuttaFehlberg
using Base.Test

config = RKFConfig((x, y) -> vecnorm(x .- y, 1), 0.0001, 1.0)

# the zero function should always return 0
for x0 in [0, 1, 2]
    @test_approx_eq rkf45_step((x, t)->0, x0, 0.0, 0.1, config)[1] 0
end

# test linear function in x
@test_approx_eq rkf45_step((x, t)->x, 0, 0.0, 0.1, config)[1] 0
@test rkf45_step((x, t)->x,  1, 0.0, 0.1, config)[1] > 0
@test rkf45_step((x, t)->x, -1, 0.0, 0.1, config)[1] < 0

# test linear function in t
@test rkf45_step((x, t)->t,  0, -1.0, 0.1, config)[1] < 0
@test rkf45_step((x, t)->t,  0,  1.0, 0.1, config)[1] > 0
@test rkf45_step((x, t)->t, -1, -1.0, 0.1, config)[1] < 0
@test rkf45_step((x, t)->t, -1,  1.0, 0.1, config)[1] > 0

# test vector input
function test_vector()
    x0 = float([1, 2, 3])
    res = rkf45_step((x, t)->reverse(x), x0, 0.0, 0.1, config)
    return typeof(res[1]) == typeof(x0)
end
@test test_vector()
