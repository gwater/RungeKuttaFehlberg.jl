# RungeKuttaFehlberg

[![Build Status](https://travis-ci.org/gwater/RungeKuttaFehlberg.jl.svg?branch=master)](https://travis-ci.org/gwater/RungeKuttaFehlberg.jl)

## Description

An implementation of the well-known Runge-Kutta-Fehlberg time integration method of 4th and 5th order (RKF45).
The algorithm integrates differential equations of the form:

    dx / dt = f[x](t)

Notably, f can be either a function or a functional of x. This is useful for certain types of partial differential equations (e.g. the heat equation).

## Usage

Import as usual:

    using RungeKuttaFehlberg

The package exports exactly one function

    rkf45_step(f, x, t, tolerance, dt[, error, safety])

which returns `dx` and `dt` as a tuple.
Most arguments should be self-explanatory but more detailed documentation is included in the package.

## Additional comments

Most likely you will iterate over `rkf45_step()` and sum up `dx` and `dt`.
The algorithm will run most efficiently if you pass the last return value for `dt` back into `rkf45_step()` at the next iteration.

The r.h.s. function `f()` must take exactly two arguments, `x` and `t`. Currently, there is no way to pass additional parameters to `f()`.
However, you can easily define an intermediate function which contains the values of each parameter and then pass it to `rkf45_step()`.

RKF45 evaluates `f()` *at least* 6 times during each step, so optimizing `f()` can increase performance a lot.

I am currently hosting this in a separate package, but I am open to suggestions w.r.t. inclusion in a package for time integration methods.
