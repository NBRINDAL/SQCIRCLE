# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 18:54:59 2025

@author: nickr
"""

# main.py
import sympy as sp
from geometry import define_line, define_circle, define_point, homothety, compute_geometry
from intersections import  line_intercepts, plot_intercepts, compute_intersection, find_all_intersections_full_parallel, collect_all_sympy_objects, compute_pair_intersection, collect_unique_points_from_full_inters_parallel
from constructibility import is_constructible_point, check_classical_constructibility, is_constructible, new_unit_coord, is_ef_point
from visualization import draw_all, show_plot, plot_intercepts, plot_zoomed_views, print_sorted_intersections, print_geometry_summary
from new_units import get_new_units_data, homothety_nu
import concurrent.futures
import matplotlib.pyplot as plt  # For plotting geometric objects
from matplotlib.patches import Circle, Rectangle, Polygon 

def main():
    # Compute the geometry
    geom = compute_geometry()

    # Define points F and E
    F = sp.Point(geom["F"][0], geom["F"][1])
    E = sp.Point(geom["E"][0], geom["E"][1])

    # The distance between F and E
    d_FE = F.distance(E)

    # Run intersection calculations
    full_inters = find_all_intersections_full_parallel(geom)
    print("\nFull sweep intersections (symbolic):")
    for key, inters in full_inters.items():
        print(f"Intersection of objects {key}:")
        for item in inters:
            print("   ", sp.pretty(item))

    # Collect unique points from intersections
    unique_pts = collect_unique_points_from_full_inters_parallel(full_inters)
    
    
    
    # List for storing matching intersections
    matching_intersections = []

# Check distances of each intersection
    for pt in unique_pts:
        distance_from_F = F.distance(pt)
    # If the distance from F to the intersection is equal to the distance from F to E
    if sp.simplify(distance_from_F - d_FE) == 0:
        matching_intersections.append(pt)

# Print out the matching intersections
    for match in matching_intersections:
        print(f"Intersection at {match} is a distance of F to E away from F")
    
    # Classify points
    ef_pts = []
    classical_pts = []
    non_classical_pts = []

    # Check classification of the unique points
    for pt in unique_pts:
        # Ensure evaluation of symbolic points before classification
        x_val = float(pt.x.evalf())
        y_val = float(pt.y.evalf())

        # Check if the point is EF
        if is_ef_point(pt, geom):
            ef_pts.append(pt)
        # Check if the point is classically constructible
        elif is_constructible_point(pt):
            classical_pts.append(pt)
        else:
            non_classical_pts.append(pt)

    # Print classifications for debugging purposes
    print("\nClassical points:")
    for pt in classical_pts:
        print(f"   {pt}")
        
    print("\nEF points:")
    for pt in ef_pts:
        print(f"   {pt}")

    print("\nNon-classical points:")
    for pt in non_classical_pts:
        print(f"   {pt}")



    # Optionally print geometry summary
    print_geometry_summary(geom)
    
    # Get the new units data
    new_units_text, F, E, k_solution, U1, U2, D1, D2, F_to_U1_new, F_to_U2_new, A_to_F_new, B_to_F_new, E_to_intercept_new, gap_bottom_new, gap_vertical_new = get_new_units_data()  # Get all values from new_units.py
    
    # --- NEW: Search for intercepts that are rational multiples of F→E ---
    # Define the extra circle centers as sympy Points
    extra_centers = [
        sp.Point(-10, 20),
        sp.Point(-10, 0),
        sp.Point(10, 0),
        sp.Point(10, 20),
        sp.Point(0, 0)
    ]
    
    # Create extra circles: each center gets circles with integer radii 1 to 12
    extra_circles = [sp.Circle(center, r) for center in extra_centers for r in range(1, 13)]
    
    # Build a list of geometric objects from your geometry dictionary.
    # (Adjust this if your geometry is stored differently.)
    all_objects = []
    for key, obj in geom.items():
        if isinstance(obj, (sp.Line, sp.Circle, sp.Segment)):
            all_objects.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                if isinstance(item, (sp.Line, sp.Circle, sp.Segment)):
                    all_objects.append(item)
    
    # Search for intersections between each extra circle and the objects in all_objects.
    rational_intercepts = []
    rational_multiples = []
    for circle in extra_circles:
        for obj in all_objects:
            # Skip comparing a circle to itself if necessary.
            if circle == obj:
                continue
            pts = circle.intersection(obj)
            for pt in pts:
                # Compute the ratio of the distance from F to pt over F→E
                ratio = sp.nsimplify(F.distance(pt) / d_FE)
                if ratio.is_Rational:
                    rational_intercepts.append(pt)
                    rational_multiples.append(ratio)
    
    # Print the rational intercepts and their factors
    print("\nIntercepts that are rational multiples of F→E:")
    for pt, factor in zip(rational_intercepts, rational_multiples):
        print(f"Point: {pt}, Factor: {factor}")
       # --- NEW: Check all computed intersections for rational multiples of F→E ---
    rational_intercepts = []
    rational_multiples = []
    
    # Iterate over every computed intersection (unique_pts)
    for pt in unique_pts:
        # Compute the ratio: distance(F, pt) / d_FE
        ratio = sp.nsimplify(F.distance(pt) / d_FE)
        # If the simplified ratio is a rational number, record the point
        if ratio.is_Rational:
            rational_intercepts.append(pt)
            rational_multiples.append(ratio)
    
    # Explicitly check and add point E, ensuring it is included
    ratio_E = sp.nsimplify(F.distance(E) / d_FE)
    if ratio_E.is_Rational and E not in rational_intercepts:
        rational_intercepts.append(E)
        rational_multiples.append(ratio_E)
    
    # Print out all intercepts that are rational multiples of F→E
    print("\nIntercepts (from full intersection search) that are rational multiples of F→E:")
    for pt, factor in zip(rational_intercepts, rational_multiples):
        print(f"Point: {pt}, Factor: {factor}")
        
    # Print the symbolic math and relevant values in the console
    print("\n=== New Units Data ===")
    print(new_units_text)
    print(f"F = {F}")
    print(f"E = {E}")
    print(f"k_solution = {k_solution}")
    print(f"U1 = {U1}, U2 = {U2}")
    print(f"D1 = {D1}, D2 = {D2}")
    print("\n=== Computed Values ===")
    print(f"F → U1 in new units: {F_to_U1_new}")
    print(f"F → U2 in new units: {F_to_U2_new}")
    print(f"A → F in new units: {A_to_F_new}")
    print(f"B → F in new units: {B_to_F_new}")
    print(f"E → Intercept in new units: {E_to_intercept_new}")
    
    # Show plot and pass the new_units_text to visualize
    show_plot(geom, classical_pts, non_classical_pts, ef_pts)  # Pass the new_units_text to plotting function

    
  


if __name__ == "__main__":
    main()
