// Gmsh project created on Fri May 30 10:14:25 2025


lc = 1e-2;
ld = 1e-3;

Point(1) = {0,0,0,lc};
Point(2) = {0.015,0,0,ld};
Point(3) = {0,0.015,0,ld};
Point(4) = {0,0.15,0,lc};
Point(5) = {0.15,0,0,lc};
Point(6) = {0.15,0.15,0,lc};

Line(1) = {4, 3};
Line(2) = {4, 6};
Line(3) = {6, 5};
Line(4) = {5, 2};

Circle(5) = {3, 1, 2};

Curve Loop(1) = {2, 3, 4, -5, -1};

Plane Surface(1) = {1};

Physical Curve("left", 1) = {1};
Physical Curve("top", 2) = {2};
Physical Curve("right", 3) = {3};
Physical Curve("bottom", 4) = {4};
Physical Surface("Plate_surf") = {1};

Save "plate_with_hole.msh";
