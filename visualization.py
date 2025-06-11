# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 18:50:59 2025

@author: nickr
"""

#visualization.py
import sympy as sp
from geometry import define_line, define_circle, define_point, homothety, compute_geometry
from intersections import line_intercepts, plot_intercepts, compute_intersection, find_all_intersections_full_parallel, collect_all_sympy_objects, compute_pair_intersection, collect_unique_points_from_full_inters_parallel
from constructibility import is_constructible_point, check_classical_constructibility, is_constructible, new_unit_coord, is_ef_point

import matplotlib.pyplot as plt  # For plotting geometric objects
from matplotlib.patches import Circle, Rectangle, Polygon  # To draw shapes on the plot
import numpy as np  # For numeric computation (e.g., arrays, ranges)

def plot_intercepts(ax, P1, P2, label_prefix=""):
    """Plots the x and y intercepts of the line defined by points P1 and P2."""
    x_int, y_int = line_intercepts(P1, P2)
    
    # Ensure both intercepts are handled as symbolic and use evalf() if needed
    if y_int != sp.S.NaN:
        ax.plot(0, float(y_int.evalf()), 'ro', markersize=8, label=label_prefix + " y-int")
    
    if x_int != sp.S.NaN:
        ax.plot(float(x_int.evalf()), 0, 'bo', markersize=8, label=label_prefix + " x-int")


def draw_all(ax, geom, classical_pts=None, non_classical_pts=None, ef_pts=None):
    # Draw flower centers
    for (x, y) in geom["flower_centers"]:
        ax.add_patch(Circle((x, y), 10, fill=False, edgecolor='blue', lw=0.5))

    # Draw the large square
    (sq_x, sq_y, sq_w, sq_h) = geom["large_square"]
    ax.add_patch(Rectangle((sq_x, sq_y), sq_w, sq_h, fill=False, edgecolor='gray', linestyle='--', lw=2))

    # Draw lines
    ax.plot(geom["line345"][:, 0], geom["line345"][:, 1], 'r-', lw=2)
    ax.plot(geom["line345_mirror"][:, 0], geom["line345_mirror"][:, 1], 'r-', lw=2)
    ax.plot(geom["line2016"][:, 0], geom["line2016"][:, 1], 'g-', lw=2)
    ax.plot(geom["line2016_mirror"][:, 0], geom["line2016_mirror"][:, 1], 'g-', lw=2)

    # Draw circles
    (c5_center, c5_rad) = geom["circle_r5"]
    ax.add_patch(Circle(c5_center, c5_rad, fill=False, edgecolor='purple', lw=2, linestyle=':'))
    (c6_center, c6_rad) = geom["circle_r6"]
    ax.add_patch(Circle(c6_center, c6_rad, fill=False, edgecolor='orange', lw=4))

    # Draw polygon (square or other shapes)
    ax.add_patch(Polygon(geom["square_diag15"], closed=True, fill=False, edgecolor='magenta', lw=4, linestyle='--'))

    # Draw points (F and E)
    F = geom["F"]
    E = geom["E"]
    ax.plot(F[0], F[1], 'bo', markersize=15)
    ax.plot(E[0], E[1], 'co', markersize=15)

    if geom["intersection_345_r5"]:
        ax.plot(geom["intersection_345_r5"][0], geom["intersection_345_r5"][1], 'ko', markersize=10)

    # Plot intersection points based on classification (if available)
    if classical_pts is not None:
        for pt in classical_pts:
            x_val = float(pt.x.evalf())
            y_val = float(pt.y.evalf())
            ax.plot(x_val, y_val, 'ro', markersize=8)  # Red for classical points

    if ef_pts is not None:
        for pt in ef_pts:
            x_val = float(pt.x.evalf())
            y_val = float(pt.y.evalf())
            ax.plot(x_val, y_val, 'o', markersize=8, color='orange')  # Green for EF points

    if non_classical_pts is not None:
        for pt in non_classical_pts:
            x_val = float(pt.x.evalf())
            y_val = float(pt.y.evalf())
            ax.plot(x_val, y_val, 'bo', markersize=8)  # Blue for non-classical points

    # Draw additional elements
    ax.add_patch(Polygon(geom["homothety_square"], closed=True, fill=False, edgecolor='darkblue', lw=4))
    
    # Draw lines with slopes
    slopes = geom["middle_lines_slopes"]
    for m_val in slopes:
        x_vals = np.linspace(-0.2, 0.2, 100)
        y_vals = [F[1] + m_val * x for x in x_vals]
        ax.plot(x_vals, y_vals, linestyle='--', lw=2, color='brown')

    # Draw circles with radii R_small and R_new
    R_small_val = float(geom["R_small"])
    ax.add_patch(Circle(F, R_small_val, fill=False, edgecolor='darkred', lw=4))
    ax.add_patch(Circle(E, R_small_val, fill=False, edgecolor='darkgreen', lw=4))

    R_new_val = float(geom["R_new"])
    ax.add_patch(Circle(E, R_new_val, fill=False, edgecolor='purple', lw=4))

    # Draw the P points and the line between P_left and P_perfect
    P_draw = geom["P_draw"]
    P_perfect = geom["P_perfect"]
    P_down = geom["P_down"]
    P_left = geom["P_left"]
    intersection_pt_small = geom["intersection_pt_small"]
    ax.plot(P_draw[0], P_draw[1], 'magenta', markersize=12)
    ax.plot(P_perfect[0], P_perfect[1], 'blue', markersize=12)
    
    t_vals = np.linspace(0, 1, 200)
    up_x = [P_left[0] + t*(P_perfect[0] - P_left[0]) for t in t_vals]
    up_y = [P_left[1] + t*(P_perfect[1] - P_left[1]) for t in t_vals]
    ax.plot(up_x, up_y, 'g-')

    B_val = geom["B_val"]
    ax.plot(np.linspace(-0.04, 0.04, 100), [float(B_val)] * 100, 'y--')

    # Draw the circles based on Delta_r
    Delta_r_val = float(geom["Delta_r"])
    theta = np.linspace(0, 2 * np.pi, 300)
    r1 = 3 * Delta_r_val
    r2 = 15 * Delta_r_val
    circle1_x = P_draw[0] + r1 * np.cos(theta)
    circle1_y = P_draw[1] + r1 * np.sin(theta)
    circle2_x = P_draw[0] + r2 * np.cos(theta)
    circle2_y = P_draw[1] + r2 * np.sin(theta)
    ax.plot(circle1_x, circle1_y, 'm-', lw=3)
    ax.plot(circle2_x, circle2_y, 'orange', lw=3)

    ax.plot(intersection_pt_small[0], intersection_pt_small[1], 'ko', markersize=8)

    # Draw rectangular grid elements (L_bridge)
    L_bridge = geom["L_bridge"]
    n_full = geom["n_full"]
    for i in range(n_full + 1):
        x0 = intersection_pt_small.x
        y0 = P_down[1] + i * L_bridge
        ax.add_patch(Rectangle((x0, y0), L_bridge, L_bridge, fill=False, edgecolor='darkorange', linestyle='--', lw=2))

    # Draw small circles at all circle centers
    for (cx, cy) in geom["all_circle_centers"]:
        ax.add_patch(Circle((cx, cy), Delta_r_val, color='lightgray', fill=False, lw=1))

    # Draw the inverted square
    inv_sq = geom["inverted_square"]
    ax.add_patch(Polygon(inv_sq, closed=True, fill=False, edgecolor='purple', linestyle='--', lw=3))
    

   

def plot_zoomed_views(geom, classical_pts, non_classical_pts, ef_pts):
    fig, axes = plt.subplots(1, 3, figsize=(60, 20))

    # Define scale ranges for different zoom levels
    scales = [
        ((-17, 17), (-9, 25), "Largest Scale"),
        ((-0.15, 0.15), (7.35, 7.66), "Middle Scale"),
        ((-0.04, 0.04), (7.46, 7.55), "Smallest Scale")
    ]

    # Loop through each scale and plot the geometry
    for ax, (xlim, ylim, title) in zip(axes, scales):
        draw_all(ax, geom, classical_pts, non_classical_pts, ef_pts)  # Pass all arguments to draw_all
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16)
        ax.grid(True)

    # Display the plots
    plt.tight_layout()
    plt.savefig('H:/circle/output_plot.pdf', format='pdf')
    plt.show()


def show_plot(geom, classical_pts, non_classical_pts, ef_pts, new_units_text=None):
    """Display the plot with intersections and geometric objects."""
   
    plot_zoomed_views(geom, classical_pts, non_classical_pts, ef_pts)  # Pass all arguments to plot_zoomed_views
    
    if new_units_text:
        plt.text(-0.19, 7.32, new_units_text, fontsize=12, color='brown',
                 bbox=dict(facecolor='white', alpha=0.8))

# After plotting, use savefig to save the image as a PDF




def print_sorted_intersections(classical_pts, non_classical_pts, ef_pts):
    """Print symbolic and numeric coordinates of intersections, classified."""
    print("\n===== CLASSICALLY CONSTRUCTIBLE INTERSECTIONS =====")
    for idx, pt in enumerate(classical_pts, start=1):
        symb = sp.pretty(pt)
        x_val = float(pt.x.evalf())
        y_val = float(pt.y.evalf())
        print(f"C{idx}: Symbolic: {symb}  |  Numeric: ({x_val:.5f}, {y_val:.5f})")
    
    print("\n===== NON-CLASSICALLY CONSTRUCTIBLE INTERSECTIONS =====")
    for idx, pt in enumerate(non_classical_pts, start=1):
        symb = sp.pretty(pt)
        x_val = float(pt.x.evalf())
        y_val = float(pt.y.evalf())
        print(f"NC{idx}: Symbolic: {symb}  |  Numeric: ({x_val:.5f}, {y_val:.5f})")

    print("\n===== EF CLASSIFICATION =====")
    for idx, pt in enumerate(ef_pts, start=1):
        symb = sp.pretty(pt)
        x_val = float(pt.x.evalf())
        y_val = float(pt.y.evalf())
        print(f"EF{idx}: Symbolic: {symb}  |  Numeric: ({x_val:.5f}, {y_val:.5f})")

def print_geometry_summary(geom):
    """
    Print a summary of key geometric values, including points and distances.
    """
    print("\n===== GEOMETRY SUMMARY =====")
    # Example geometry summary
    A_val = geom["A_val"]
    B_val = geom["B_val"]
    Delta_pi = geom["Delta_pi"]
    Delta_r = geom["Delta_r"]
    P_draw = geom["P_draw"]
    P_perfect = geom["P_perfect"]
    P_down = geom["P_down"]
    P_left = geom["P_left"]
    intersection_pt_small = geom["intersection_pt_small"]
    L_bridge = geom["L_bridge"]
    vertical_gap = geom["vertical_gap"]
    n_exact = geom["n_exact"]
    n_full = geom["n_full"]
    error_below = geom["error_below"]
    error_above = geom["error_above"]
    top_sq_bl = geom["top_sq_bl"]
    top_sq_tr = geom["top_sq_tr"]
    top_center = geom["top_center"]

    print(f"P_draw:               {P_draw}")
    print(f"P_down:               {P_down}")
    print(f"P_left:               {P_left}")
    print(f"P_perfect:            {P_perfect}")
    print(f"Intersection (small scale): {intersection_pt_small}")
    print(f"Steeper ∩ Left:       {geom['steeper_left_int_val']}")
    print(f"Steeper ∩ Bridging:   {intersection_pt_small}")
    print(f"Δπ = B - A:           {Delta_pi:.10f}")
    print(f"Δr = Δπ / 9:          {Delta_r:.10f}")
    print(f"L_bridge:             {L_bridge:.10f} units")
    print(f"Vertical gap from P_down to P_perfect: {vertical_gap:.10f} units")
    print(f"Exact number of squares needed: {n_exact:.10f} (i.e. {n_full} full squares plus extra {n_exact - n_full:.10f})")
    print(f"Error if stacking {n_full} squares (under P_perfect): {error_below:.10f} units")
    print(f"Error if stacking {n_full+1} squares (over P_perfect):  {error_above:.10f} units")
    print(f"Top square bottom-left: {top_sq_bl}")
    print(f"Top square top-right:   {top_sq_tr}")
    print(f"Top square center (X center): {top_center}")