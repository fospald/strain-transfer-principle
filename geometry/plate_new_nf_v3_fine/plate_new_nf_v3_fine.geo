SetFactory("OpenCASCADE");

Merge "plate_new_nf_v3_fine.step";
Coherence;

// http://transit.iut2.upmf-grenoble.fr/cgi-bin/info2www?(gmsh)Specifying+mesh+element+sizes
Mesh.CharacteristicLengthFromCurvature = 1;
Mesh.MinimumCirclePoints = 20;
Mesh.CharacteristicLengthMax = 3;
//Mesh.CharacteristicLengthMin = 3;
Mesh.CharacteristicLengthFactor = 0.3;


//+
Physical Volume("matrix") = {2};
//+
Physical Volume("fiber") = {6, 3, 4, 5};
//+
Physical Surface("left") = {34, 12, 1, 28, 20};
//+
Physical Surface("right") = {35};
//+
Physical Surface("top") = {38, 27, 32, 11, 19};
//+
Physical Surface("bottom") = {33};
//+
Physical Surface("back") = {36};
//+
Physical Surface("front") = {37};
//+
Physical Surface("inlet") = {12, 1, 28, 20};
//+
Physical Surface("outlet") = {27, 32, 11, 19};
//+
Physical Curve("fiber_center") = {19, 13, 7};
//+
Physical Curve("fiber_center") += {7, 13, 19};


