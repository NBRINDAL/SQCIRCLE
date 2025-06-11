# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 18:48:53 2025

@author: nickr
"""

# constructibility.py
import sympy as sp
from geometry import define_line, define_circle, define_point, homothety, compute_geometry
import concurrent.futures

def is_constructible_point(pt):
    """Checks if a point is classically constructible."""
    return pt.x.is_rational and pt.y.is_rational

def check_classical_constructibility(geom):
    """Check for constructibility of points in geometry."""
    constructible_points = []
    for pt_name, pt in geom.items():
        if isinstance(pt, sp.Point):
            if is_constructible_point(pt):
                constructible_points.append(pt_name)
    return constructible_points
def is_constructible(expr):
    """Check if an expression is constructible."""
    try:
        poly = sp.minimal_polynomial(expr, sp.Symbol('x'))
        deg = sp.degree(poly)
        while deg % 2 == 0 and deg > 1:
            deg //= 2
        return deg == 1
    except Exception:
        return False

def classify_points_parallel(unique_pts, geom, max_workers=6):
    def classify(pt):
        return {
            'pt': pt,
            'is_constructible': is_constructible_point(pt),
            'is_ef': is_ef_point(pt, geom)
        }

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(classify, unique_pts))
    
    return results



def new_unit_coord(pt, geom):
    """Convert point coordinates to new unit coordinates."""
    k_val = geom["k"]
    new_x = sp.simplify(pt.x / (sp.Rational(5, 2) * k_val))
    new_y = sp.simplify((pt.y - sp.Rational(15, 2)) / (sp.Rational(5, 2) * k_val))
    return new_x, new_y


def is_ef_point(pt, geom, tol=1e-12):
    """Check if a point is EF-based (rational in new unit coordinates)."""
    new_x, new_y = new_unit_coord(pt, geom)
    return new_x.is_Rational and new_y.is_Rational

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