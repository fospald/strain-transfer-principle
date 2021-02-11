import meshio
from dolfin import * 

filename = "plate_new_nf_v3_fine.msh"

msh = meshio.read(filename)

meshio.write_points_cells("plate_new_nf_v3_fine.vtk", msh.points, msh.cells)
meshio.write("plate_new_nf_v3_fine.xdmf", meshio.Mesh(points=msh.points, cells={"tetra": msh.cells["tetra"]}))

meshio.write("plate_new_nf_v3_fine_paths.xdmf",
                meshio.Mesh(points=msh.points,
                cells={"line": msh.cells["line"]},
                cell_data={"line": {"ids": msh.cell_data["line"]["gmsh:physical"]}}))

meshio.write("plate_new_nf_v3_fine_boundaries.xdmf",
                meshio.Mesh(points=msh.points,
                cells={"triangle": msh.cells["triangle"]},
                cell_data={"triangle": {"ids": msh.cell_data["triangle"]["gmsh:physical"]}}))

meshio.write("plate_new_nf_v3_fine_subdomains.xdmf",
                meshio.Mesh(points=msh.points,
                cells={"tetra": msh.cells["tetra"]},
                cell_data={"tetra": {"ids": msh.cell_data["tetra"]["gmsh:physical"]}}))

mesh = Mesh()
with XDMFFile("plate_new_nf_v3_fine.xdmf") as infile:
    infile.read(mesh)

mesh_paths = MeshValueCollection("size_t", mesh, 1) 
with XDMFFile("plate_new_nf_v3_fine_paths.xdmf") as infile:
    infile.read(mesh_paths, "ids")

mesh_boundaries = MeshValueCollection("size_t", mesh, 2) 
with XDMFFile("plate_new_nf_v3_fine_boundaries.xdmf") as infile:
    infile.read(mesh_boundaries, "ids")

mesh_subdomains = MeshValueCollection("size_t", mesh, 3) 
with XDMFFile("plate_new_nf_v3_fine_subdomains.xdmf") as infile:
    infile.read(mesh_subdomains, "ids")

File("plate_new_nf_v3_fine.pvd").write(mesh)
File("plate_new_nf_v3_fine_paths.pvd").write(cpp.mesh.MeshFunctionSizet(mesh, mesh_paths))
File("plate_new_nf_v3_fine_boundaries.pvd").write(cpp.mesh.MeshFunctionSizet(mesh, mesh_boundaries))
File("plate_new_nf_v3_fine_subdomains.pvd").write(cpp.mesh.MeshFunctionSizet(mesh, mesh_subdomains))

