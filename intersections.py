# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 18:41:15 2025

@author: nickr
"""

#intersections.py
import sympy as sp
from geometry import define_line, define_circle, define_point, homothety, compute_geometry
import concurrent.futures

# --- 1. Helper Functions for Intercepts and Basic Intersections --- #

def line_intercepts(P1, P2):
    x, y = sp.symbols('x y', real=True)
    x1, y1 = map(sp.sympify, P1)
    x2, y2 = map(sp.sympify, P2)
    if x2 == x1:
        return (sp.S.NaN, sp.S.NaN)
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m*x1
    x_int = -b/m if m != 0 else sp.S.NaN
    return (x_int, b)

def plot_intercepts(ax, P1, P2, label_prefix=""):
    """Plots the x and y intercepts of the line defined by points P1 and P2."""
    x_int, y_int = line_intercepts(P1, P2)
    if y_int != sp.S.NaN:
        ax.plot(0, float(y_int.evalf()), 'ro', markersize=8, label=label_prefix + " y-int")
    if x_int != sp.S.NaN:
        ax.plot(float(x_int.evalf()), 0, 'bo', markersize=8, label=label_prefix + " x-int")

# --- 2. Intersection Computation --- #
def compute_intersection(obj1, obj2):
    """Computes the intersection of two geometric objects."""
    try:
        inter = obj1.intersection(obj2)
    except Exception:
        inter = []
    return inter
def find_all_intersections(geom):
    """Find all intersections for the geometry."""
    intersections = {}

    # Define lines and circle based on geometry
    left_line = define_line(geom["P_left"], geom["P_draw"])
    right_line = define_line(geom["P_perfect"], geom["P_left"])
    circle = define_circle(geom["P_draw"], geom["Delta_r"])

    intersections["left_line_circle"] = compute_intersection(left_line, circle)
    intersections["right_line_circle"] = compute_intersection(right_line, circle)

    return intersections

def find_all_intersections_parallel(geom, max_workers=10):
    sp_objects = []
    sp_objects.append(geom["left_line_small"])
    sp_objects.append(geom["right_line_small"])
    sp_objects.append(geom["upward_line_small"])
    sp_objects.append(geom["steeper_line"])
    P_draw = sp.Point(geom["P_draw"])
    Delta_r = sp.sympify(geom["Delta_r"])
    circle1_sym = sp.Circle(P_draw, 3*Delta_r)
    circle2_sym = sp.Circle(P_draw, 15*Delta_r)
    sp_objects.append(circle1_sym)
    sp_objects.append(circle2_sym)
    
    # Parallelize the computation of intersections between objects
    pairs = [(i, j, sp_objects[i], sp_objects[j]) for i in range(len(sp_objects)) for j in range(i + 1, len(sp_objects))]
    
    def compute_pair_intersection(pair):
        i, j, obj_i, obj_j = pair
        try:
            inter = obj_i.intersection(obj_j)
        except Exception:
            inter = []
        return (i, j, inter)
    
    intersections = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, j, inter in executor.map(compute_pair_intersection, pairs):
            if inter:
                intersections[(i, j)] = inter
    return intersections

def collect_unique_points(inters_dict):
    """Collects unique points from intersections."""
    unique_pts = []
    for inter_list in inters_dict.values():
        for item in inter_list:
            if isinstance(item, sp.Point):
                x_val = float(item.x.evalf())
                y_val = float(item.y.evalf())
                if not any(
                    abs(float(pt.x.evalf()) - x_val) < 1e-12 and abs(float(pt.y.evalf()) - y_val) < 1e-12
                    for pt in unique_pts
                ):
                    unique_pts.append(item)
    return unique_pts

# --- 3. Parallel Intersection Computation (Full Sweep) --- #
def collect_all_sympy_objects(geom):
    """Collect all symbolic geometric objects."""
    objs = []
    for center in geom["flower_centers"]:
        objs.append(sp.Circle(sp.Point(center), 10))
    
    sq = geom["large_square"]
    pts_sq = [sp.Point(sq[0], sq[1]), sp.Point(sq[0] + sq[2], sq[1]), 
              sp.Point(sq[0] + sq[2], sq[1] + sq[3]), sp.Point(sq[0], sq[1] + sq[3])]
    objs.append(sp.Polygon(*pts_sq))

    objs.append(sp.Line(sp.Point(-10, 15), sp.Point(10, 0)))
    objs.append(sp.Line(sp.Point(10, 15), sp.Point(-10, 0)))
    objs.append(sp.Line(sp.Point(-10, 20), sp.Point(6, 0)))
    objs.append(sp.Line(sp.Point(10, 20), sp.Point(-6, 0)))
    objs.append(sp.Circle(sp.Point(5, 0), 5))
    objs.append(sp.Circle(sp.Point(0, 0), 6))
    
    pts_diag = [sp.Point(0, 7.5), sp.Point(-7.5, 0), sp.Point(0, -7.5), sp.Point(7.5, 0)]
    objs.append(sp.Polygon(*pts_diag))
    objs.append(sp.Polygon(*[sp.Point(p[0], p[1]) for p in geom["homothety_square"]]))

    F = sp.Point(geom["F"])
    for m in geom["middle_lines_slopes"]:
        objs.append(sp.Line(F, sp.Point(1, F.y + m)))

    objs.append(sp.Circle(sp.Point(geom["F"]), geom["R_small"]))
    objs.append(sp.Circle(sp.Point(geom["E"]), geom["R_small"]))
    objs.append(sp.Circle(sp.Point(geom["E"]), geom["R_new"]))
    objs.append(sp.Line(sp.Point(geom["P_left"]), sp.Point(geom["P_perfect"])))
    
    B_val = geom["B_val"]
    objs.append(sp.Line(sp.Point(-1, B_val), sp.Point(1, B_val)))
    objs.append(sp.Circle(sp.Point(geom["P_draw"]), 3 * geom["Delta_r"]))
    objs.append(sp.Circle(sp.Point(geom["P_draw"]), 15 * geom["Delta_r"]))
    
    objs.append(geom["left_line_small"])
    objs.append(geom["right_line_small"])
    objs.append(geom["upward_line_small"])
    objs.append(geom["steeper_line"])

    for center in geom["all_circle_centers"]:
        objs.append(sp.Circle(sp.Point(center), geom["Delta_r"]))

    for i in range(geom["n_full"] + 1):
        x0 = geom["intersection_pt_small"].x
        y0 = geom["P_down"][1] + i * geom["L_bridge"]
        pts_rect = [sp.Point(x0, y0),
                    sp.Point(x0 + geom["L_bridge"], y0),
                    sp.Point(x0 + geom["L_bridge"], y0 + geom["L_bridge"]),
                    sp.Point(x0, y0 + geom["L_bridge"])]
        objs.append(sp.Polygon(*pts_rect))
    objs.append(geom["inverted_square"])
    
    return objs

def compute_pair_intersection(pair):
    """Computes intersection for a pair of objects."""
    i, j, obj_i, obj_j = pair
    try:
        inter = obj_i.intersection(obj_j)
    except Exception:
        inter = []
    return (i, j, inter)

def find_all_intersections_full_parallel(geom, max_workers=10):
    """Find all intersections for the full sweep of geometry."""
    objs = collect_all_sympy_objects(geom)  # Collect all symbolic objects
    pairs = []
    
    # Collect all pairs of objects to compute intersections
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            pairs.append((i, j, objs[i], objs[j]))

    intersections = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, j, inter in executor.map(compute_pair_intersection, pairs):
            if inter:
                intersections[(i, j)] = inter
    return intersections


def process_intersection(inter_list):
    """Process intersections to collect unique points."""
    unique_pts = []
    for item in inter_list:
        if isinstance(item, sp.Point):
            x_val = float(item.x.evalf())
            y_val = float(item.y.evalf())
            if not any(
                abs(float(pt.x.evalf()) - x_val) < 1e-12 and abs(float(pt.y.evalf()) - y_val) < 1e-12
                for pt in unique_pts
            ):
                unique_pts.append(item)
    return unique_pts

def collect_unique_points_from_full_inters_parallel(full_inters, max_workers=10):
    """Collect unique points from all full intersections using parallel processing."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        result = list(executor.map(process_intersection, full_inters.values()))
    
    # Flatten the results and return the unique points
    all_pts = [pt for sublist in result for pt in sublist]
    return all_pts

# --- 4. Constructibility and EF Classification --- #
def is_constructible(expr):
    """Checks if a point is constructible."""
    try:
        poly = sp.minimal_polynomial(expr, sp.Symbol('x'))
        deg = sp.degree(poly)
        while deg % 2 == 0 and deg > 1:
            deg //= 2
        return deg == 1
    except Exception:
        return False

def is_constructible_point(pt):
    """Check if a point is constructible."""
    return is_constructible(pt.x) and is_constructible(pt.y)

def new_unit_coord(pt, geom):
    """Converts point coordinates to new unit system."""
    k_val = geom["k"]
    new_x = sp.simplify(pt.x / (sp.Rational(5, 2) * k_val))
    new_y = sp.simplify((pt.y - sp.Rational(15, 2)) / (sp.Rational(5, 2) * k_val))
    return new_x, new_y

def is_ef_point(pt, geom, tol=1e-12):
    """Checks if a point is EF (rational coordinates in the new unit system)."""
    new_x, new_y = new_unit_coord(pt, geom)
    return new_x.is_Rational and new_y.is_Rational


