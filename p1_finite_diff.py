cdimport numpy as np
import math
import matplotlib.pyplot as plt

def f(x):

    """
    Function to compute the value of e^(-x^2).

    Parameters:
    x (float): Input value.

    Returns:
    float: Computed value of e^(-x^2).
    """

    return math.exp((-x ** 2))

def f_first_finite_diff(f, x, h):
    """
    Computes the first finite difference approximation of the derivative of f at x.

    Parameters:
    f (function): Function whose derivative is to be approximated.
    x (float): Point at which the derivative is to be approximated.
    h (float): Step size.

    Returns:
    float: First finite difference approximation of the derivative of f at x.
    """
    return ((f(x+h) - f(x-h)) / (2 * h))

def f_second_finite_diff(f, x, h):
    """
    Computes the second finite difference approximation of the second derivative of f at x.

    Parameters:
    f (function): Function whose second derivative is to be approximated.
    x (float): Point at which the second derivative is to be approximated.
    h (float): Step size.

    Returns:
    float: Second finite difference approximation of the second derivative of f at x.
    """
    return ((f(x+h) - (2 * f(x)) + f(x-h)) / (h ** 2))

def f_prime_analytical(x):
    """
    Computes the analytical first derivative of f at x.

    Parameters:
    x (float): Point at which the first derivative is to be computed.

    Returns:
    float: Analytical first derivative of f at x.
    """
    return (-2*x*math.exp(-x ** 2))

def f_double_prime_analytical(x):
    """
    Computes the analytical second derivative of f at x.

    Parameters:
    x (float): Point at which the second derivative is to be computed.

    Returns:
    float: Analytical second derivative of f at x.
    """
    return (4*x*math.exp(-x ** 2))-(2*math.exp(-x ** 2))

x = 6

# given stability region (10^-4 to 10^-8)
h = 10 ** -5

print(f"First Finite Difference: {f_first_finite_diff(f, x, h)}")
print(f"Second Finite Difference: {f_second_finite_diff(f, x, h)}")

print(f"First Prime Analytical: {f_prime_analytical(x)}")
print(f"Second Prime Analytical: {f_double_prime_analytical(x)}")


h_values = np.logspace(-15, 0, 100)
errors_first_derivative = []
errors_second_derivative = []

for h in h_values:

    # derivatives
    f_prime_num = f_first_finite_diff(f, x, h)
    f_double_prime_num = f_second_finite_diff(f, x, h)
    f_prime_ana = f_prime_analytical(x)
    f_double_prime_ana = f_double_prime_analytical(x)

    # abs errors
    error_first = np.abs(f_prime_num - f_prime_ana)
    error_second = np.abs(f_double_prime_num - f_double_prime_ana)

    # Store errors
    errors_first_derivative.append(error_first)
    errors_second_derivative.append(error_second)

# Plot the results
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors_first_derivative, label="First Derivative Error", marker=".")
plt.loglog(h_values, errors_second_derivative, label="Second Derivative Error", marker="x")
plt.axvline(x=1e-8, color="red", linestyle="--", label="Optimal h (approx)")
plt.xlabel("Step Size (h)")
plt.ylabel("Error")
plt.title("Error in Numerical Derivatives vs. Step Size (h)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()