# SQCIRCLE
Python Squaring the Circle
# Squaring the Circle — Dynamic Construction

This project implements a symbolic geometric construction inspired by classical squaring the circle, using a dynamic and multi-scalar approach. It introduces motion, homotopy scaling, and layered construction logic using a radius-6 circle, 3-4-5 triangles, phi slopes, and a (16 run / 20 rise) proportional triangle system.

## 🧠 Concept

The design was originally drafted in Rhino and tested using symbolic Python. The method doesn’t aim to "square the circle" in the classical sense (which is impossible with compass and straightedge), but instead demonstrates a geometric system that moves through a configuration where the square and circle areas match precisely.

## 📁 Files

- `main.py` — Entry point. Runs the full symbolic construction and visualization.
- `constructibility.py`, `geometry.py`, `intersections.py`, `visualization.py`, `new_units.py` — Helper modules defining geometry, logic, and plotting.

## ▶️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/NBRINDAL/SQCIRCLE.git
   cd SQCIRCLE
## 🔄 Construction Animation

This animated sequence shows the live progression of the geometric method. By layering transformations, proportions, and radial forms, it illustrates how the system dynamically aligns square and circle areas.

![Geometric Construction Animation](geometric_construction_animation.gif)
