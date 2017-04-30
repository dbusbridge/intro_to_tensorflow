import numpy as np


# Linear gradient
def linear(x, bias, gradient, noise_sd=0.01):
    noise = np.random.normal(scale=noise_sd, size=x.size)
    return x * gradient + bias + noise


# Quadratic gradient
def quadratic(x, bias, gradient, noise_sd=0.01):
    noise = np.random.normal(scale=noise_sd, size=x.size)
    return x * gradient + bias + noise


# Lambda
def general(x, f, noise_sd=0.01):
    noise = np.random.normal(scale=noise_sd, size=x.size)
    return f(x) + noise
