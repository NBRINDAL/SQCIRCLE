# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 11:59:15 2025

@author: nickr
"""

#new_units.py
import sympy as sp
from sympy import sqrt, pi, Rational
import mpmath

# --- 1) Symbolic Setup & Homothety --- 
mpmath.mp.prec = 300
k = sp.Symbol('k', real=True)
F = sp.Point(0, Rational(15, 2))       # F = (0, 7.5)
center_initial = sp.Point(0, Rational(10, 1))  # (0, 10)
A_initial = sp.Point(-10, 20)
B_initial = sp.Point(-10, 0)
C_initial = sp.Point(10, 0)
D_initial = sp.Point(10, 20)

# Homothety function to scale points
def homothety_nu(P, k_val):
    return sp.Point((1 - k_val)*F.x + k_val*P.x,
                    (1 - k_val)*F.y + k_val*P.y)

# Find scale factor 'k' such that distance from F to E becomes 1 unit
y_center_expr = (1 - k)*F.y + k*center_initial.y
target_expr = 3*sqrt(2*pi)  # The target value for E's y-coordinate
eq = sp.Eq(y_center_expr, target_expr)
k_solution = sp.solve(eq, k)[0]  # k = (3√(2π) - 7.5) / 2.5

# Apply homothety transformation to all points
A = homothety_nu(A_initial, k_solution)
B = homothety_nu(B_initial, k_solution)
C_pt = homothety_nu(C_initial, k_solution)
D = homothety_nu(D_initial, k_solution)
square_corners = [A, B, C_pt, D]

# Rename the square's center to E
E = sp.Point(0, y_center_expr.subs(k, k_solution))
d_FE = 2.5 * k_solution  # F→E in original scale
scale_factor = 1/d_FE  # New unit = F→E in original scale

# --- 2) Define Six Lines Through F ---
slopes = {
    "345_up":    sp.Rational(3,4),
    "345_down":  -sp.Rational(3,4),
    "2016_up":   sp.Rational(5,4),
    "2016_down": -sp.Rational(5,4),
    "45_up":     sp.Integer(1),
    "45_down":   -sp.Integer(1),
}
lines = {name: sp.Line(F, sp.Point(1, F.y + m)) for name, m in slopes.items()}

# --- 3) Define Points U1, U2, D1, D2 ---
U1 = sp.Point(-7.5*k_solution, 7.5 - 7.5*k_solution)   # bottom intercept from 45° up
D2 = sp.Point(7.5*k_solution, 7.5 - 7.5*k_solution)    # bottom intercept from 45° down
U2 = sp.Point(10*k_solution, 7.5 + 10*k_solution)        # right intercept from 45° up
D1 = sp.Point(-10*k_solution, 7.5 + 10*k_solution)       # left intercept from 45° down

# --- 4) Additional Intercepts ---
G = sp.Point(-10*k_solution, 7.5 + 7.5*k_solution)  # "345_down" with left edge
H = sp.Point(10*k_solution, 7.5 + 7.5*k_solution)   # "345_up" with right edge
J = sp.Point(-6*k_solution, 7.5 - 7.5*k_solution)   # "2016_up" with bottom edge
K = sp.Point(6*k_solution, 7.5 - 7.5*k_solution)    # "2016_down" with bottom edge

# --- 5) New Unit System (F→E = 1) ---
d_FE_val = float(d_FE.evalf(50))
scale_factor = 1/d_FE_val  # New unit = F→E in original scale

# For reference, expected new units (these are the same units as in the new_units_text):
F_to_U1_new = sp.sqrt(F.distance(U1)) * scale_factor
F_to_U2_new = sp.sqrt(F.distance(U2)) * scale_factor
A_to_F_new  = sp.sqrt(A.distance(F)) * scale_factor
B_to_F_new  = sp.sqrt(B.distance(F)) * scale_factor
E_to_intercept_new = sp.sqrt(sp.Point(E.x, E.y).distance(U1)) * scale_factor
gap_bottom_new = (15 * k_solution) * scale_factor
gap_vertical_new = (20 * k_solution) * scale_factor

# --- 6) Return the New Units Text ---
new_units_text = (
    f"New Units (F→E = 1):\n"
    f"F→E = 1\n"
    f"E→Intercept = 5\n"
    f"U1-D2 = 6\n"
    f"D1-U2 = 8\n"
    f"Diagonal (side intercepts) = 5√2 ≈ 7.07\n"
    f"F→U1 = 3√2 ≈ 4.24, F→U2 = 4√2 ≈ 5.66\n"
    f"B→F = 5, A→F = √41 ≈ 6.40\n"
    f"(B→F + F→E + E→D1 = 11)\n"
    f"G,H => 345 lines intercept\n"
    f"J,K => 2016 lines intercept\n"
    f"Circles (r=5 new) at F & E\n"
    f"New circle at E (r from 2016 lines) = 4/5 new units"
)

# Return the symbolic values for later use in visualization
def get_new_units_data():
    return new_units_text, F, E, k_solution, U1, U2, D1, D2, F_to_U1_new, F_to_U2_new, A_to_F_new, B_to_F_new, E_to_intercept_new, gap_bottom_new, gap_vertical_new