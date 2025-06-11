# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 18:40:12 2025

@author: nickr
"""

#geometry.py
import sympy as sp
import numpy as np
from sympy import sqrt, pi, Rational

import mpmath

# Increase precision for symbolic calculations
mpmath.mp.prec = 300

def define_point(x, y):
    """ Define a symbolic point using given x and y coordinates. """
    return sp.Point(x, y)

def define_line(P1, P2):
    """ Define a line passing through two points. """
    return sp.Line(P1, P2)

def define_circle(center, radius):
    """ Define a circle with a given center and radius. """
    return sp.Circle(center, radius)

def define_polygon(*points):
    """ Define a polygon using a list of points. """
    return sp.Polygon(*points)

def homothety(P, F, k_val):
    """ Apply homothety transformation to point P using scaling factor k_val relative to point F. """
    return sp.Point((1 - k_val) * F.x + k_val * P.x, (1 - k_val) * F.y + k_val * P.y)

# Function to compute new unit coordinates
def new_unit_coord(pt, k_val):
    """ Convert the coordinates to new units (based on F→E distance). """
    new_x = sp.simplify(pt.x / (sp.Rational(5, 2) * k_val))
    new_y = sp.simplify((pt.y - sp.Rational(15, 2)) / (sp.Rational(5, 2) * k_val))
    return new_x, new_y

def compute_tangent_circles():
    """
    Compute all four centers for circles of radius R = EF that are tangent
    to both line345 (3x+4y-30=0) and line2016 (5x+4y-30=0).
    Return a list of (x_center, y_center, R_symbolic), with each center 
    satisfying one of the four sign combinations:
       sign345 * 5R = (3x + 4y - 30)
       sign2016 * sqrt(41)*R = (5x + 4y - 30)
    """

    x, y = sp.symbols('x y', real=True)
    
    # Radius: R = EF = 3*sqrt(2*pi) - 7.5
    R = 3*sp.sqrt(2*sp.pi) - 7.5

    # Line 345 in standard form => 3x + 4y - 30=0
    # If tangent "above" that line => 3x+4y-30 = +5R
    # If tangent "below" that line => 3x+4y-30 = -5R
    # Similarly for line2016 => 5x + 4y -30=0,
    # tangent => 5x+4y-30 = ± sqrt(41)*R

    eq_345 = 3*x + 4*y - 30
    eq_2016 = 5*x + 4*y - 30

    # We'll solve for each sign combination
    sign_combos = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
    circles = []
    
    for s345, s2016 in sign_combos:
        # Build the system:
        #   eq_345 = s345 * 5 * R
        #   eq_2016 = s2016 * sqrt(41) * R
        eqs = [
            sp.Eq(eq_345, s345 * 5 * R),
            sp.Eq(eq_2016, s2016 * sp.sqrt(41) * R)
        ]
        
        # Solve for (x, y)
        sol = sp.solve(eqs, (x, y), dict=True)
        if sol:
            # Each system is linear => 1 unique solution
            sol_x = sol[0][x]
            sol_y = sol[0][y]
            
            # Store the result along with sign info if desired
            circles.append({
                "signs": (s345, s2016),
                "x_sym": sol_x,
                "y_sym": sol_y,
                "R": R  # the symbolic R
            })
    return circles


def compute_geometry():
    geom = {}
    # Flower centers
    d = 10
    n = 2
    flower_centers = []
    for q in range(-n, n+1):
        for r in range(max(-n, -q-n), min(n, -q+n)+1):
            x = d * (q + r/2)
            y = d * (np.sqrt(3)/2 * r)
            flower_centers.append((x, y))
    geom["flower_centers"] = flower_centers
    geom["large_square"] = (-10, 0, 20, 20)
    geom["line345"] = np.array([[-10, 15], [10, 0]])
    geom["line345_mirror"] = np.array([[10, 15], [-10, 0]])
    geom["line2016"] = np.array([[-10, 20], [6, 0]])
    geom["line2016_mirror"] = np.array([[10, 20], [-6, 0]])
    geom["circle_r5"] = ((5, 0), 5)
    geom["circle_r6"] = ((0, 0), 6)
    geom["square_diag15"] = [(0, 7.5), (-7.5, 0), (0, -7.5), (7.5, 0)]
    
    # Define F and E
    F = (0, sp.Rational(15,2))      # (0,7.5)
    E = (0, 3*sp.sqrt(2*sp.pi))       # (0, 3√(2π))
    geom["F"] = F
    geom["E"] = E

    # Example intersection of line345 and circle (r=5)
    t = sp.symbols('t', real=True)
    P1 = sp.Matrix([-10, 15])
    P2 = sp.Matrix([20, 0])
    P_t = P1 + t*(P2 - P1)
    eq_345_r5 = sp.Eq((P_t[0]-5)**2 + (P_t[1])**2, 25)
    sols = sp.solve(eq_345_r5, t)
    intersection_points = []
    for sol in sols:
        xx = float(P_t[0].subs(t, sol))
        yy = float(P_t[1].subs(t, sol))
        intersection_points.append((xx, yy))
    selected_pt = None
    for pt_ in intersection_points:
        dist_ = np.sqrt((pt_[0]-10)**2 + (pt_[1]-0)**2)
        if abs(dist_ - 8) < 1e-3:
            selected_pt = pt_
            break
    geom["intersection_345_r5"] = selected_pt

    # Homothety square (using positive k)
    k = (E[1] - F[1]) / (10 - F[1])
    geom["k"] = k
    
    def homothety(P, k_val):
        return ((1 - k_val)*F[0] + k_val*P[0],
                (1 - k_val)*F[1] + k_val*P[1])
    A_init = (-10, 20)
    B_init = (-10, 0)
    C_init = (10, 0)
    D_init = (10, 20)
    A_m = homothety(A_init, k)
    B_m = homothety(B_init, k)
    C_m = homothety(C_init, k)
    D_m = homothety(D_init, k)
    geom["homothety_square"] = [A_m, B_m, C_m, D_m]
    slopes_middle = [3/4, -3/4, 5/4, -5/4, 1, -1]
    geom["middle_lines_slopes"] = slopes_middle
    R_small = 12.5 * k
    geom["R_small"] = R_small
    R_new = (sp.Rational(4,5)) * (E[1] - F[1])
    geom["R_new"] = R_new

    A_val = sp.Rational(15,2)
    B_val = 3*sp.sqrt(2*sp.pi)
    Delta_pi = B_val - A_val
    Delta_r = Delta_pi / 9
    geom["A_val"] = A_val
    geom["B_val"] = B_val
    geom["Delta_pi"] = Delta_pi
    geom["Delta_r"] = Delta_r

    P_draw = F
    P_perfect = E
    P_down = (0, A_val - 3*Delta_r)
    P_left = (-4*Delta_r, A_val - 3*Delta_r)
    geom["P_draw"] = P_draw
    geom["P_perfect"] = P_perfect
    geom["P_down"] = P_down
    geom["P_left"] = P_left

    left_line_small = sp.Line(P_draw, P_left)
    right_line_small = sp.Line(P_draw, (4*Delta_r, A_val - 3*Delta_r))
    upward_line_small = sp.Line(P_left, P_perfect)
    intersection_pt_small = upward_line_small.intersection(right_line_small)[0]
    geom["intersection_pt_small"] = intersection_pt_small
    steeper_line = sp.Line((0, A_val), (1, A_val+1.25))
    steeper_left_int = steeper_line.intersection(left_line_small)[0]
    geom["steeper_left_int_val"] = steeper_left_int

    L_bridge = intersection_pt_small.y - P_down[1]
    vertical_gap = B_val - P_down[1]
    n_exact = vertical_gap / L_bridge
    n_full = int(np.floor(float(n_exact)))
    error_below = vertical_gap - n_full * L_bridge
    error_above = (n_full + 1)*L_bridge - vertical_gap
    geom["L_bridge"] = L_bridge
    geom["vertical_gap"] = vertical_gap
    geom["n_exact"] = n_exact
    geom["n_full"] = n_full
    geom["error_below"] = error_below
    geom["error_above"] = error_above

    top_square_index = n_full
    top_square_bottom = P_down[1] + top_square_index * L_bridge
    top_square_top = top_square_bottom + L_bridge
    top_square_left = intersection_pt_small.x
    top_square_right = top_square_left + L_bridge
    top_center = (top_square_left + L_bridge/2, top_square_bottom + L_bridge/2)
    geom["top_sq_bl"] = (top_square_left, top_square_bottom)
    geom["top_sq_tr"] = (top_square_right, top_square_top)
    geom["top_sq_tl"] = (top_square_left, top_square_top)
    geom["top_sq_br"] = (top_square_right, top_square_bottom)
    geom["top_center"] = top_center

    centers_up = [(0.0, A_val + i*Delta_r) for i in range(1,10)]
    centers_down = [(0.0, A_val - i*Delta_r) for i in range(1,4)]
    centers_left = [(-j*Delta_r, A_val - 3*Delta_r) for j in range(1,5)]
    all_circle_centers = centers_up + centers_down + centers_left
    geom["all_circle_centers"] = all_circle_centers

    geom["left_line_small"] = left_line_small
    geom["right_line_small"] = right_line_small
    geom["upward_line_small"] = upward_line_small
    geom["steeper_line"] = steeper_line

     # Calculate center of the square
    center_y = F[1] - (1/3) * (E[1] - F[1])  # Y-coordinate of the center
    center_x = 0  # X-coordinate is fixed at 0
    square_center = (center_x, center_y)

    # Define the side length of the square
    side_length = (E[1] - center_y) * 2  # You can adjust this value

    # Calculate the distance from the center to the top edge (half of the side length)
    half_side = side_length / 2

    # Define the top edge of the square (y-coordinate is fixed as E.y)
    top_edge_y = E[1]

    # Calculate the bottom edge of the square
    bottom_edge_y = top_edge_y - side_length

    # Now we can calculate the four corners of the square:
    top_left = sp.Point(square_center[0] - half_side, top_edge_y)
    top_right = sp.Point(square_center[0] + half_side, top_edge_y)
    bottom_left = sp.Point(square_center[0] - half_side, bottom_edge_y)
    bottom_right = sp.Point(square_center[0] + half_side, bottom_edge_y)

    # Store the square corners in the geometry dictionary
    geom["inverted_square"] = [top_left, top_right, bottom_right, bottom_left]
    
    
    
 
    return geom

