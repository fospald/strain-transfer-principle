# strain-transfer-principle
Supplemental material for the publication "An Extension of the Strain Transfer Principle for Fiber Reinforced Materials" [https://arxiv.org/abs/2010.05857]

## Requirements

The following Python3-libraries are required:
* [fibergen](https://github.com/fospald/fibergen)
* [FEniCS 2019.1](https://fenicsproject.org/download/)


## Reproduction of results from publication

In order to reproduce the results from the publication you have to run the following commands:
1. Run matrix material homogenization for 50% fiber volume fraction using fibergen:
```bash
cd homogenization
python3 run_hom.py
```
should create the `C_hom_50.npy` file containing the Voigt matrix of the homogenized material.
2. Run calculation of strain transfer matrices (for different sensor fiber orientations) using fibergen and the homogenized matrix material from the previous step:
```bash
python3 run_stp.py
```
should create the `T_50_*.npy` files containing the strain transfer matrix for the sensor fiber angle specified in the filename (*).
3. Run the solver for the elasticity problem to create the various results from the publiaction:
```bash
cd ../solver
python3 solver.py
```
should create the result table `results/plate_new_nf_v3_fine_results_fiber_f_50_xxy.csv` (among two other csv files).


## Contents of the result CSV file

The result file contains the following columns:
- 1. i: the point index along the sensor fiber center
- 2. ei: the element index corresponding to the point in the mesh
- 3. s: the acr-length parameter of the curve corrsponding to the point
- 4. iv1: the first vertex index in the mesh for the edge containing the point
- 5. iv2: the second vertex index in the mesh for the edge containing the point
- 6-8: dx,dy,dz: the normalized direction of the edge (v2-v1)
- 9-11: umx, umy, umz: average nodal displacement of the edge (u1+u2)/2
- 12-14: dux, duy, duz: difference between nodal displacements (u2-u1)
- 15: ed: displacement in direction d (scalar product (du, d))
- 16-21: ef11, ef22, ef33, ef23, ef13, ef12: strain computed using the extended strain-transfer priciple at the edge center point
- 22-28: enf11, enf22, enf33, enf23, enf13, enf12: no-fiber solution strain at the edge center point
- 29-34: er11, er22, er33, er23, er13, er12: reference solution strain at the edge center point
- 35-40: eTnf11, eTnf22, eTnf33, eTnf23, eTnf13, eTnf12: strain computed using classical strain-transfer principle for the no-fiber solution at the edge center point

The arc-length parameter and the strain components are the basis for the plots in the publication.

