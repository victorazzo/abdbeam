# Abdbeam : composites cross section analysis
A Python package for the cross section analysis of composite material beams of any shape. 

## Main features

These are a few things you can do with **Abdbeam** today:

* Use a fast, analytical thin-walled anisotropic composite beam theory including closed cells, open branches, shear connectors and booms (discrete stiffeners containing axial and torsional stiffnesses);
* Recover replacement stiffnesses (EA, EIyy, EIzz, EIyz, GJ) and/or a full 4 x 4 stiffness matrix for beams with arbitrary layups and shapes;
* Recover centroid and shear center locations;
* Obtain internal load distributions (Nx, Nxy, Mx, My, Mxy for segments; Px and Tx for booms) for a large number of cross section load cases (defined by Px, My, Mz, Tz, Vy and Vz section loads).

Note: the effects of shear deformation and restrained warping are assumed negligible in **Abdbeam**. Check our theory section for references containing more details.

## Installing

Install using PyPI ([Python package index](https://pypi.org/project/abdbeam)) :

```sh
pip install abdbeam
```

The source code is hosted on GitHub at https://github.com/victorazzo/abdbeam

## Dependencies

- [NumPy](https://www.numpy.org)
- [Pandas](https://pandas.pydata.org)

## Example

Let's use **Abdbeam** to extract the basic properties of the cross section with two closed cells below:

![Cross Section Example](https://user-images.githubusercontent.com/24232637/50049830-266c6380-00bc-11e9-808d-97896d3f3e16.png)  

An example script to achieve this is:

```python
import abdbeam as ab
sc = ab.Section()
# Create a materials dictionary:
mts = dict()
mts[1] = ab.Laminate()
ply_mat = ab.PlyMaterial(0.166666, 148000, 9650, 4550, 0.3)
mts[1].ply_materials[1] = ply_mat
mts[1].plies = [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1],
               [45,1], [45,1], [45,1], [45,1], [45,1], [45,1]]
# Create a points dictionary based on Y and Z point coordinates:
pts = dict()
pts[1] = ab.Point(0, -35)
pts[2] = ab.Point(-50, -35)
pts[3] = ab.Point(-50, 35)
pts[4] = ab.Point(0, 35)
pts[5] = ab.Point(50, 35)
pts[6] = ab.Point(50, -35)
# Create a segments dictionary referencing point and material ids:
sgs = dict()
sgs[1] = ab.Segment(1,2,1)
sgs[2] = ab.Segment(2,3,1)
sgs[3] = ab.Segment(3,4,1)
sgs[4] = ab.Segment(4,1,1)
sgs[5] = ab.Segment(4,5,1)
sgs[6] = ab.Segment(5,6,1)
sgs[7] = ab.Segment(6,1,1)
# Point the dictionaries to the section
sc.materials = mts
sc.points = pts
sc.segments = sgs
# Calculate and output section properties
sc.calculate_properties()
sc.summary()
```
Which prints:
```sh
Section Summary
===============

Number of points: 6
Number of segments: 7
Number of cells: 2

Centroid
--------
yc = -0.26778063610746794
zc = 0.0

Shear Center
------------
ys = -0.15909143196852882
zs = -0.0005864193651285987

Replacement Stiffnesses
-----------------------
EA = 68032952.29455881
EIyy = 52483433984.419556
EIzz = 83640874772.70999
EIyz = 0.0
GJ = 12376231660.76024

[P_c] - Beam Stiffness Matrix at the Centroid
---------------------------------------------
[[ 6.80329523e+07  0.00000000e+00  2.46320132e+05 -1.43701515e+08]
 [ 0.00000000e+00  5.24834340e+10  0.00000000e+00  0.00000000e+00]
 [ 2.46320132e+05  0.00000000e+00  8.36408748e+10 -2.12142163e+07]
 [-1.43701515e+08  0.00000000e+00 -2.12142163e+07  1.23762317e+10]]

[W_c] - Beam Compliance Matrix at the Centroid
----------------------------------------------
[[1.50683149e-08 0.00000000e+00 1.66286490e-28 1.74959530e-10]
 [0.00000000e+00 1.90536313e-11 0.00000000e+00 0.00000000e+00]
 [1.57282135e-25 0.00000000e+00 1.19558821e-11 2.04936911e-14]
 [1.74959530e-10 0.00000000e+00 2.04936911e-14 8.28315446e-11]]

[P] - Beam Stiffness Matrix at the Origin
-----------------------------------------
[[ 6.80329523e+07  0.00000000e+00 -1.79715871e+07 -1.43701515e+08]
 [ 0.00000000e+00  5.24834340e+10  0.00000000e+00  0.00000000e+00]
 [-1.79715871e+07  0.00000000e+00  8.36456213e+10  1.72662667e+07]
 [-1.43701515e+08  0.00000000e+00  1.72662667e+07  1.23762317e+10]]

[W] - Beam Compliance Matrix at the Origin
------------------------------------------
[[1.50691722e-08 0.00000000e+00 3.20155371e-12 1.74965018e-10]
 [0.00000000e+00 1.90536313e-11 0.00000000e+00 0.00000000e+00]
 [3.20155371e-12 0.00000000e+00 1.19558821e-11 2.04936911e-14]
 [1.74965018e-10 0.00000000e+00 2.04936911e-14 8.28315446e-11]]
```

Now let's create two load cases (identified as 101 and 102)  and calculate the internal loads:

```python
sc.loads = dict()
sc.loads[101] = ab.Load(1000.0,25000,-36000)
sc.loads[102] = ab.Load(15000)
sc.calculate_internal_loads()
```
And to print all internal loads:
```python
sc.print_internal_loads()
```

## License

BSD-3

## Contribute

**Abdbeam** is at its early development stages and we encourage you to pitch in and [contribute on GitHub](https://github.com/victorazzo/abdbeam). Guidelines for contributors are in the works, so stay tuned.

## Theory

For the theory behind Abdbeam, the most complete reference is:

[Victorazzo DS, De Jesus A. A Kollár and Pluzsik anisotropic composite beam theory for arbitrary multicelled cross sections. Journal of Reinforced Plastics and Composites. 2016 Dec;35(23):1696-711.](https://journals.sagepub.com/doi/abs/10.1177/0731684416665493)

These are also great references on its originating theory:

* [ Kollár LP and Springer GS. Mechanics of composite structures. Cambridge: Cambridge University Press, 2003.](https://www.amazon.com/Mechanics-Composite-Structures-L%C3%A1szl%C3%B3-Koll%C3%A1r/dp/0521126908/ref=sr_1_1?ie=UTF8&qid=1544936929&sr=8-1&keywords=Mechanics+of+composite+structures)
* [Kollár LP and Pluzsik A. Analysis of thin-walled composite beams with arbitrary layup. J Reinf Plast Compos 2002; 21: 1423–1465.](https://journals.sagepub.com/doi/abs/10.1177/0731684402021016928)