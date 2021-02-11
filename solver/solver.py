
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import petsc4py
from petsc4py import PETSc

petsc4py.init(sys.argv)
np.set_printoptions(precision=4, suppress=True)

# Test for PETSc
if not has_linear_algebra_backend("PETSc"):
	print("DOLFIN has not been configured with PETSc. Exiting.")
	exit()

# Set backend to PETSC
parameters["linear_algebra_backend"] = "PETSc"


def build_nullspace(V, x):
	"""Function to build null space for 3D elasticity"""

	# Create list of vectors for null space
	nullspace_basis = [x.copy() for i in range(6)]

	# Build translational null space basis
	V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
	V.sub(1).dofmap().set(nullspace_basis[1], 1.0);
	V.sub(2).dofmap().set(nullspace_basis[2], 1.0);

	# Build rotational null space basis
	V.sub(0).set_x(nullspace_basis[3], -1.0, 1);
	V.sub(1).set_x(nullspace_basis[3],  1.0, 0);
	V.sub(0).set_x(nullspace_basis[4],  1.0, 2);
	V.sub(2).set_x(nullspace_basis[4], -1.0, 0);
	V.sub(2).set_x(nullspace_basis[5],  1.0, 1);
	V.sub(1).set_x(nullspace_basis[5], -1.0, 2);

	for x in nullspace_basis:
		x.apply("insert")

	# Create vector space basis and orthogonalize
	basis = VectorSpaceBasis(nullspace_basis)
	basis.orthonormalize()

	return basis

def mandelVecToMat(V):
	s = 2**-0.5
	return np.array([
		[V[0], V[5]*s, V[4]*s],
		[V[5]*s, V[1], V[3]*s],
		[V[4]*s, V[3]*s, V[2]]
	])

def voigtVecToMat(V):
	s = 1.0
	return np.array([
		[V[0], V[5]*s, V[4]*s],
		[V[5]*s, V[1], V[3]*s],
		[V[4]*s, V[3]*s, V[2]]
	])


def matToVoigtVec(M):
	return np.array([M[0,0], M[1,1], M[2,2], M[1,2], M[0,2], M[0,1]])

def mandelMatrixToTensor(C):
	C = np.copy(C)
	C[3:6,0:3] /= 2**0.5
	C[0:3,3:6] /= 2**0.5
	C[3:6,3:6] /= 2
	return voigtMatrixToTensor(C)

def tensorToMandelMatrix(T):
	# 11 22 33 23 13 12
	C = tensorToVoigtMatrix(T)
	C[3:6,0:3] *= 2**0.5
	C[0:3,3:6] *= 2**0.5
	C[3:6,3:6] *= 2
	return C

def voigtMatrixToTensor(C):
	# 11 22 33 23 13 12
	# 0  1  2  3  4  5
	V = np.array([
		[0, 5, 4],
		[5, 1, 3],
		[4, 3, 2]])
	T = np.zeros((3, 3, 3, 3));
	for k in range(3):
		for l in range(3):
			for s in range(3):
				for t in range(3):
					T[k,l,s,t] = C[V[k,l],V[s,t]]
	return T

def tensorToVoigtMatrix(T):
	# 11 22 33 23 13 12
	I = np.array([0, 1, 2, 1, 0, 0])
	J = np.array([0, 1, 2, 2, 2, 1])
	C = np.zeros((6, 6));
	for k in range(6):
		for l in range(6):
			C[k, l] = T[I[k], J[k], I[l], J[l]]
	return C

def rotateTensor(T, R):
	RT = np.zeros_like(T)
	for k in range(3):
		for l in range(3):
			for s in range(3):
				for t in range(3):
					for m in range(3):
						for n in range(3):
							for p in range(3):
								for r in range(3):
									RT[k,l,s,t] += T[m,n,p,r] * R[k, m] * R[l, n] * R[s, p] * R[t, r]
	return RT

def rotateX(alpha):
	s = np.sin(alpha)
	c = np.cos(alpha)
	return np.array([
		[1, 0, 0],
		[0, c, -s],
		[0, s,  c]
	]);

def rotateY(alpha):
	s = np.sin(alpha)
	c = np.cos(alpha)
	return np.array([
		[ c, 0, s],
		[ 0, 1, 0],
		[-s, 0, c]
	]);

def rotateZ(alpha):
	s = np.sin(alpha)
	c = np.cos(alpha)
	return np.array([
		[c, -s, 0],
		[s,  c, 0],
		[0,  0, 1]
	]);

# swap x and z component of voigt matrix
def swap_xz(M):
	# before swap: 11 22 33 23 13 12
	# after swap:  33 22 11 12 13 23
	M = np.copy(M)
	M[[0, 2],:] = M[[2, 0],:]
	M[:,[0, 2]] = M[:,[2, 0]]
	M[[3, 5],:] = M[[5, 3],:]
	M[:,[3, 5]] = M[:,[5, 3]]
	return M

# swap y and z component of voigt matrix
def swap_yz(M):
	# before swap: 11 22 33 23 13 12
	# after swap:  11 33 22 23 12 13
	M = np.copy(M)
	M[[1, 2],:] = M[[2, 1],:]
	M[:,[1, 2]] = M[:,[2, 1]]
	M[[4, 5],:] = M[[5, 4],:]
	M[:,[4, 5]] = M[:,[5, 4]]
	return M

def swap_none(M):
	M = np.copy(M)
	return M

# init problem data
geometry = "plate_new_nf_v3_fine"
volfrac = 50

r_f = 2.0	# fiber radius
E_f = 73.0	# fiber E-modulus

sample_fact = 2
T_path = None
coating_subdomain_id = None
fiber_path_id = None

if geometry == "plate_new_nf_v3_fine":
	R = 20
	pull_direction = (1.0, 0.0, 0.0)
	reinforcement_dir = "x"
	fiber_dir1 = "x"
	fiber_dir2 = "y"
	fiber_path_id = 11
	r_f = 1.0
	fiber_subdomain_id = 2
	if 0:
		# left right
		fixed_boundary_id = 3
		pull_boundary_id = 4
	elif 0:
		# bottom top
		fixed_boundary_id = 6
		pull_boundary_id = 5
		pull_direction = (0.0, 1.0, 0.0)
	else:
		# bottom top shear
		fixed_boundary_id = 6
		pull_boundary_id = 5
		pull_direction = (1.0, 0.0, 0.0)

	t1 = np.linspace(0, 1, 60*sample_fact)
	t2 = np.linspace(0, 0.5*np.pi, int(0.5*np.pi*R*sample_fact))[1:-1]
	t3 = np.linspace(0, 1, 60*sample_fact)
	sample_path = [(t*60, 20, 5) for t in t1] + [(60 + R*np.sin(t), 40 - R*np.cos(t), 5) for t in t2] + [(80, 40 + t*60, 5) for t in t1]
	fiber_angle = [0.0 for t in t1] + [t for t in t2] + [0.5*np.pi for t in t1]
	# handle special case for which we can compute the transfer matrices in the arc
	if reinforcement_dir == "x" and fiber_dir1 == "x" and fiber_dir2 == "y":
		T_path = np.zeros((len(fiber_angle), 6, 6))
		for i in range(len(fiber_angle)):
			alpha = fiber_angle[i]
			alpha_deg = alpha*180/np.pi
			alpha_deg0 = int(alpha_deg)
			alpha_deg1 = min(alpha_deg0 + 1, 90)
			# fiber direction is in x afer swap_xz
			T0 = np.loadtxt("../homogenization/T_%d_zx%d.npy" % (volfrac, alpha_deg0))
			T0 = swap_xz(T0)
			T1 = np.loadtxt("../homogenization/T_%d_zx%d.npy" % (volfrac, alpha_deg1))
			T1 = swap_xz(T1)
			phi = alpha_deg - alpha_deg0
			T = (1-phi)*T0 + phi*T1
			# rotate fiber direction around z-axis to have angle alpha with x-axis
			# for T_zx0.np the reinforcement is in z direction (x direction afer swap_xz)
			# for T_zx90.np the reinforcement is in y direction (y direction afer swap_xz)
			# afer rotation by 90 degree around the z-axis it is aligned again in x-direction
			T_path[i] = tensorToMandelMatrix(rotateTensor(mandelMatrixToTensor(T), rotateZ(alpha)))
else:
	raise "unknown geometry %s" % geometry

if reinforcement_dir == "x":
	swap_coords_C = swap_xz
elif reinforcement_dir == "y":
	swap_coords_C = swap_yz
elif reinforcement_dir == "z":
	swap_coords_C = swap_none

def get_T_params(fiber_dir):
	if fiber_dir == reinforcement_dir:
		T_file = "zz"
		swap_coords_T = swap_coords_C
	elif fiber_dir == "x" and reinforcement_dir == "y":
		T_file = "yz"
		swap_coords_T = swap_xz
	elif fiber_dir == "x" and reinforcement_dir == "z":
		T_file = "xz"
		swap_coords_T = swap_xz
	elif fiber_dir == "y" and reinforcement_dir == "x":
		T_file = "xz"
		swap_coords_T = swap_yz
	elif fiber_dir == "y" and reinforcement_dir == "z":
		T_file = "yz"
		swap_coords_T = swap_yz
	elif fiber_dir == "z" and reinforcement_dir == "x":
		T_file = "xz"
		swap_coords_T = swap_none
	elif fiber_dir == "z" and reinforcement_dir == "y":
		T_file = "yz"
		swap_coords_T = swap_none
	return (T_file, swap_coords_T)

T1_file, swap_coords_T1 = get_T_params(fiber_dir1)
T2_file, swap_coords_T2 = get_T_params(fiber_dir2)

# Load mesh from file
print("load mesh")
geometry_dir = "../geometry"
result_dir = "results"
mesh = Mesh()
with XDMFFile(MPI.comm_world, "%s/%s/%s.xdmf" % (geometry_dir, geometry, geometry)) as infile:
	infile.read(mesh)

mesh_paths_file = "%s/%s/%s_paths.xdmf" % (geometry_dir, geometry, geometry)
if os.path.exists(mesh_paths_file):
	mesh_paths = MeshValueCollection("size_t", mesh, 2) 
	with XDMFFile(MPI.comm_world, mesh_paths_file) as infile:
		infile.read(mesh_paths, "ids")
	mesh_paths = cpp.mesh.MeshFunctionSizet(mesh, mesh_paths)
else:
	mesh_paths = None

mesh_boundaries = MeshValueCollection("size_t", mesh, 2) 
with XDMFFile(MPI.comm_world, "%s/%s/%s_boundaries.xdmf" % (geometry_dir, geometry, geometry)) as infile:
	infile.read(mesh_boundaries, "ids")
mesh_boundaries = cpp.mesh.MeshFunctionSizet(mesh, mesh_boundaries)

mesh_subdomains = MeshValueCollection("size_t", mesh, 3) 
with XDMFFile(MPI.comm_world, "%s/%s/%s_subdomains.xdmf" % (geometry_dir, geometry, geometry)) as infile:
	infile.read(mesh_subdomains, "ids")
mesh_subdomains = cpp.mesh.MeshFunctionSizet(mesh, mesh_subdomains)


class StiffnessTensor(UserExpression):
	def __init__(self, materials, Cm, Cf, Cc, **kwargs):
		self.materials = materials
		self.Cm = Cm.ravel()
		self.Cc = Cc.ravel()
		self.Cf = Cf.ravel()
		super().__init__(**kwargs)
	def eval_cell(self, values, x, cell):
		if self.materials[cell.index] == fiber_subdomain_id:
			# the fiber
			values[:] = self.Cf
		elif self.materials[cell.index] == coating_subdomain_id:
			# the fiber
			values[:] = self.Cc
		else:
			# the matrix
			values[:] = self.Cm
	def value_shape(self):
		return (6,6)


# postfix: postfix for all result filenames
# Cm: stiffness matrix for matrix material
# Cm: stiffness matrix for fiber material
# T:  strain transfer matrix applied to FBG strains for the result table field t11, t22, ...
# sample_path: list of points for sampling the strains in the results
# strain_nf: strain field without fiber or None
def solve_stuff(postfix, Cm, Cf, Cc, T, sample_path, strain_nf, strain_r, fiber_edges, r_f, E_f):

	C = StiffnessTensor(mesh_subdomains, Cm, Cf, Cc, degree=0)

	# strain computation
	def epsilon(u):
		return as_vector([
			u[0].dx(0),
			u[1].dx(1),
			u[2].dx(2),
			u[1].dx(2) + u[2].dx(1),
			u[0].dx(2) + u[2].dx(0),
			u[0].dx(1) + u[1].dx(0)
		])

		# return sym(grad(u))

	def vec_to_mat(v, s=1):
		return as_matrix([
			[  v[0], s*v[5], s*v[4]],
			[s*v[5],   v[1], s*v[3]],
			[s*v[4], s*v[3],   v[2]]])

	# Stress computation
	def sigma(eps):
		return dot(C, eps)

	print("solve_stuff", postfix)

	# Create function space
	V = VectorFunctionSpace(mesh, "Lagrange", 1)

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	a = dot(sigma(epsilon(u)), epsilon(v))*dx
	f = Constant((0.0, 0.0, 0.0))
	L = inner(f, v)*dx

	# Set up boundary condition on inner surface
	c = Constant((0.0, 0.0, 0.0))
	bc1 = DirichletBC(V, c, mesh_boundaries, fixed_boundary_id)	# left boundary
	c = Constant(pull_direction)
	bc2 = DirichletBC(V, c, mesh_boundaries, pull_boundary_id)	# right boundary
	bcs = [bc1, bc2]

	# Assemble system, applying boundary conditions and preserving
	# symmetry)
	print("assemble_system")
	A, b = assemble_system(a, L, [])

	if not fiber_edges is None:
		# modify system matrix
		Am = as_backend_type(A).mat()
		Am.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
		bv = as_backend_type(b).vec()

		# vertex to dof map
		v2d = vertex_to_dof_map(V)

		# compute matrix compliance in Mandel notation
		Sm = np.copy(Cm)
		Sm[3:6,0:3] *= 2**0.5
		Sm[0:3,3:6] *= 2**0.5
		Sm[3:6,3:6] *= 2
		Sm = np.linalg.inv(Sm)
		SmT = mandelMatrixToTensor(Sm)

		boundary_values = {}
		#for bc in bcs:
		#	boundary_values.update(bc.get_boundary_values())

		# integrate fiber streching term
		print("streching term")
		for ei in fiber_edges:
			edge = Edge(mesh, ei)
			v1, v2 = vertices(edge)
			p1 = v1.point().array()
			p2 = v2.point().array()
			l = np.linalg.norm(p2 - p1)
			d = (p2 - p1)/l

			# compute matrix E-modulus in direction d
			E_m = 1.0/np.einsum('ijkl,i,j,k,l', SmT, d, d, d, d)
			D = (E_f - E_m)*np.pi*(r_f**2)

			# get dofs for each vertex
			iv1 = v1.global_index()
			iv2 = v2.global_index()
			dofs1 = v2d[(3*iv1):(3*iv1 + 3)]
			dofs2 = v2d[(3*iv2):(3*iv2 + 3)]

			Dl = D/l

			# add contribution to system matrix
			# Dl * (u2x*dx+u2y*dy+u2z*dz - u1x*dx+u1y*dy+u1z*dz) * (v2x*dx+v2y*dy+v2z*dz - v1x*dx+v1y*dy+v1z*dz)

			for i in range(2):
				dofs_u = [dofs2, dofs1][i]
				sign_u = 1 - 2*i
				for j in range(2):
					dofs_v = [dofs2, dofs1][j]
					sign_v = 1 - 2*j
					for k in range(3):
						for p in range(3):
							"""
							u_hasbc = dofs_u[k] in boundary_values
							v_hasbc = dofs_v[p] in boundary_values
							if u_hasbc and v_hasbc:
								# no contribution since v=0
								pass
							elif u_hasbc:
								# add to rhs
								u_val = boundary_values[dofs_u[k]]
								bv.setValue(dofs_v[p], -u_val*Dl*sign_u*sign_v*d[k]*d[p],
									addv=PETSc.InsertMode.ADD)
							elif v_hasbc:
								# no contribution since v=0
								pass
							else:
							"""
							Am.setValue(dofs_u[k], dofs_v[p], Dl*sign_u*sign_v*d[k]*d[p],
								addv=PETSc.InsertMode.ADD)

		# integrate fiber bending term
		print("bending term")
		for ei in range(len(fiber_edges)-1):

			# TODO: remove
			break

			ei1 = fiber_edges[ei]
			ei2 = fiber_edges[ei+1]
			edge1 = Edge(mesh, ei1)
			edge2 = Edge(mesh, ei2)
			v11, v12 = vertices(edge1)
			v21, v22 = vertices(edge2)

			# find common vertex
			if v11.index() == v21.index():
				v1 = v12
				v = v11
				v2 = v22
			elif v11.index() == v22.index():
				v1 = v12
				v = v11
				v2 = v21
			elif v12.index() == v21.index():
				v1 = v11
				v = v12
				v2 = v22
			elif v12.index() == v22.index():
				v1 = v11
				v = v12
				v2 = v21

			p1 = v1.point().array()
			p = v.point().array()
			p2 = v2.point().array()
			l1 = np.linalg.norm(p1 - p)
			l2 = np.linalg.norm(p2 - p)
			l = np.linalg.norm(p2 - p1)
			d1 = (p1 - p)/l1
			d2 = (p2 - p)/l2
			d = (p2 - p1)/l
			ds = 0.5*(l1 + l2)

			# compute matrix E-modulus in direction d
			E_m = 1.0/np.einsum('ijkl,i,j,k,l', SmT, d, d, d, d)
			EI = 1.0*(E_f - E_m)*np.pi/4.0*(r_f**4)

			# get dofs for each vertex
			iv1 = v1.global_index()
			iv = v.global_index()
			iv2 = v2.global_index()
			dofs1 = v2d[(3*iv1):(3*iv1 + 3)]
			dofs = v2d[(3*iv):(3*iv + 3)]
			dofs2 = v2d[(3*iv2):(3*iv2 + 3)]

			# phi1 = (I - d1 o d1) (u1 - u) x d1 / l1
			# phi2 = (I - d2 o d2) (u2 - u) x d2 / l2
			# dphi = phi1 - phi2
			# kappa = dphi/ds
			# I = pi/4*r_f**4
			# Energy = 1/2*E*I*kappa**2*ds
			# Variational form:
			# dEnergy/du[v] = E*I*dot(dphi(u), dphi(v))/ds

			# define cross product matrices for x d1 and x d2
			"""
			W1 = np.array([
				[0.0, -d1[2], d1[1]],
				[d1[2], 0.0, -d1[0]],
				[-d1[1], d1[0], 0.0]
			])
			W2 = np.array([
				[0.0, -d2[2], d2[1]],
				[d2[2], 0.0, -d2[0]],
				[-d2[1], d2[0], 0.0]
			])
			"""

			W1 = np.array([
				[0.0, -d[2], d[1]],
				[d[2], 0.0, -d[0]],
				[-d[1], d[0], 0.0]
			])
			W2 = -W1

			# define matrices I - d1 o d1
			#P1 = np.eye(3) - np.outer(d1, d1)
			#P2 = np.eye(3) - np.outer(d2, d2)
			P1 = np.eye(3) - np.outer(d, d)
			P2 = P1

			# gather matrix and corss products, s.t.
			# phi1 = S1 (u1 - u)
			# phi2 = S2 (u2 - u)
			S1 = np.dot(P1, W1) / l1
			S2 = np.dot(P2, W2) / l2
			dS = S2 - S1

			# gather all constants into single constant
			D = EI/ds

			# Variational form:
			# dEnergy/du[v] = D*dot(dphi(u),dphi(v))
			# dEnergy/du[v] = D*dot(S1(u1 - u) - S2(u2 - u), S1(v1 - v) - S2(v2 - v))
			# dEnergy/du[v] = D*dot(S1 u1 + (S2 - S1) u - S2 u2, S1 v1 + (S2 - S1) v - S2 v2)

			for i in range(3): # first argument term of dot product
				S_u = [S1, dS, -S2][i]
				dofs_u = [dofs1, dofs, dofs2][i]
				for j in range(3): # second argument term of dot product
					S_v = [S1, dS, -S2][j]
					dofs_v = [dofs1, dofs, dofs2][j]
					for k in range(3): # index in u
						for p in range(3): # index in v
							"""
							u_hasbc = dofs_u[k] in boundary_values
							v_hasbc = dofs_v[p] in boundary_values
							if u_hasbc and v_hasbc:
								# no contribution since v=0
								pass
							elif u_hasbc:
								# add to rhs
								u_val = boundary_values[dofs_u[k]]
								bv.setValue(dofs_v[p], -u_val*D*np.dot(S_u[:,k], S_v[:,p]),
									addv=PETSc.InsertMode.ADD)
							elif v_hasbc:
								# no contribution since v=0
								pass
							else:
							"""
							Am.setValue(dofs_u[k], dofs_v[p], D*np.dot(S_u[:,k], S_v[:,p]),
								addv=PETSc.InsertMode.ADD)
		
		Am.assemble()
	
	
	# apply Dirichlet boundary conditions, if we modified by "accident"
	for bc in bcs:
		bc.zero_columns(A, b, 1.0)	# symmetrically apply boundary conditions again (this also sets b to the bcs and makes the row identity)
	
	
	# Create solution function
	u = Function(V)

	# Create near null space basis (required for smoothed aggregation
	# AMG). The solution vector is passed so that it can be copied to
	# generate compatible vectors for the nullspace.
	null_space = build_nullspace(V, u.vector())

	# Attach near nullspace to matrix
	as_backend_type(A).set_near_nullspace(null_space)

	# Create PETSC smoothed aggregation AMG preconditioner and attach near
	# null space
	pc = PETScPreconditioner("petsc_amg")

	# Use Chebyshev smoothing for multigrid
	PETScOptions.set("mg_levels_ksp_type", "chebyshev")
	PETScOptions.set("mg_levels_pc_type", "jacobi")

	# Improve estimate of eigenvalues for Chebyshev smoothing
	PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
	PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

	# Create CG Krylov solver and turn convergence monitoring on
	solver = PETScKrylovSolver("cg", pc)
	solver.parameters["monitor_convergence"] = True

	# Set matrix operator
	solver.set_operator(A);

	# Compute solution
	print("solve")
	solver.solve(u.vector(), b);

	# Save solution to VTK format
	print("save solution")
	File("%s/%s_u_%s.pvd" % (result_dir, geometry, postfix), "compressed") << u

	# Save colored mesh partitions in VTK format if running in parallel
	if MPI.size(mesh.mpi_comm()) > 1:
		File("%s/%s_partitions_%s.pvd" % (result_dir, geometry, postfix), "compressed") << MeshFunction("size_t", mesh, mesh.topology().dim(), MPI.rank(mesh.mpi_comm()))

	# Project and write stress field to post-processing file
	W = TensorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
	#W = TensorFunctionSpace(mesh, "Lagrange", 1)
	
	strain = project(vec_to_mat(epsilon(u), 0.5), V=W, solver_type="cg")
	eps_filename = "%s/%s_epsilon_%s.pvd" % (result_dir, geometry, postfix)
	File(eps_filename, "compressed") << strain
	print("writing", eps_filename)

	stress = project(vec_to_mat(sigma(epsilon(u))), V=W, solver_type="cg")
	sigma_filename = "%s/%s_sigma_%s.pvd" % (result_dir, geometry, postfix)
	File(sigma_filename, "compressed") << stress
	print("writing", sigma_filename)

	def voigt_vec(M):
		return [M[0], M[4], M[8], M[5], M[2], M[1]]

	f = open("%s/%s_results_%s.csv" % (result_dir, geometry, postfix), "wt+")
	#f.write("\t".join(["i", "x", "y", "z", "ux", "uy", "uz", "e11", "e22", "e33", "e23", "e13", "e12", "s11", "s22", "s33", "s23", "s13", "s12", "et11", "et22", "et33", "et23", "et13", "et12"] + ["t%d%d" % (i, j) for j in range(6) for i in range(6)]) + "\n")
	f.write("\t".join(["i", "s", "x", "y", "z", "ux", "uy", "uz", "e11", "e22", "e33", "e23", "e13", "e12", "s11", "s22", "s33", "s23", "s13", "s12", "et11", "et22", "et33", "et23", "et13", "et12", "dx", "dy", "dz", "ed"]) + "\n")
	i = 0
	s = 0.0
	print("sampling path")
	for pi in sample_path:
		ui = u(*pi)
		ei = strain(*pi)
		si = stress(*pi)

		if i+1 < len(sample_path):
			v = np.array(sample_path[i+1]) - np.array(pi)
		elif i > 0:
			v = np.array(pi) - np.array(sample_path[i-1])
		else:
			raise "len(sample_path) < 2"
		
		l = np.linalg.norm(v)
		d = v/l

		# strain in fiber direction
		ed = np.dot(np.dot(ei.reshape((3, 3)), d), d)

		"""
		if strain_nf is None:
			W = np.eye(6)
		else:
			# TODO: need 6 loadcases to compute the strain transfer matrix
			# ei_nf = voigt_vec(strain_nf(*pi))
			W = np.eye(6)
			# ei = T*ei_nf
			# ei*inv(ei_nf) = T*ei_nf
		"""

		ei = voigt_vec(ei)
		si = voigt_vec(si)

		eim = np.copy(ei)
		eim[3:6] *= 2**0.5	# to Mandel notation
		eti = np.matmul(T[i], eim)
		eti[3:6] /= 2**0.5	# back to Voigt notation

		f.write("\t".join(map(str, [i, s, *pi, *ui, *ei, *si, *eti, *d, ed])) + "\n")
		# f.write("\t".join([i, *pi, *ui, *ei, *si, [W[i,j] for j in range(6) for i in range(6)]]) + "\n")

		s += l
		i += 1

	f.close()

	print("sampling edges")
	if not fiber_edges is None:
		# write strain in fiber direction and strain from extended strain transfer principle

		f = open("%s/%s_results_fiber_%s.csv" % (result_dir, geometry, postfix), "wt+")

		# index, edge index, current path length at edge center, vertex1 index, vertex2 index, fiber direction vector, displacement difference between v1 and v2, mean displacement, strain in fiber direction
		f.write("\t".join(["i", "ei", "s", "iv1", "iv2", "dx", "dy", "dz", "umx", "umy", "umz", "dux", "duy", "duz", "ed", "ef11", "ef22", "ef33", "ef23", "ef13", "ef12", "enf11", "enf22", "enf33", "enf23", "enf13", "enf12", "er11", "er22", "er33", "er23", "er13", "er12", "eTnf11", "eTnf22", "eTnf33", "eTnf23", "eTnf13", "eTnf12"]) + "\n")

		# TODO: multiprocessing
		u_vec = u.vector().get_local()

		i = 0
		s = 0.0
		for ei in fiber_edges:
			edge = Edge(mesh, ei)
			v1, v2 = vertices(edge)
			p1 = v1.point().array()
			p2 = v2.point().array()
			l = np.linalg.norm(p2 - p1)
			d = (p2 - p1)/l
			
			iv1 = v1.global_index()
			iv2 = v2.global_index()
			dofs1 = v2d[(3*iv1):(3*iv1 + 3)]
			dofs2 = v2d[(3*iv2):(3*iv2 + 3)]
			u1 = u_vec[dofs1]
			u2 = u_vec[dofs2]
			print("u1, u2", u1, u2)
			du = u2 - u1
			um = 0.5*(u1 + u2)
			ed = np.dot(du, d)/l

						# find closest strain transfer matrix
			pm = (p1 + p2)/2
			imin = 0
			dmin = 1e100
			for iT, pi in enumerate(sample_path):
				di = np.linalg.norm(np.array(pi) - pm)
				if di < dmin:
					imin = iT
					dmin = di
			
			# compute full strain tensor inside fiber from extended strain transfer principle
			ei_r = strain_r(*pm)
			ei_r = voigt_vec(ei_r)
			ei_nf = strain_nf(*pm)
			ei_nf = voigt_vec(ei_nf)
			eim_nf = np.copy(ei_nf)
			eim_nf[3:6] *= 2**0.5	# to Mandel notation
			P = np.outer(d, d)
			#print("eim_nf", eim_nf)
			#print("T", T[imin])
			#print("P", P)
			#print("ed", ed)
			#print(np.eye(3))
			ei_Tnf = np.matmul(T[imin], eim_nf)
			ei_Tnf[3:6] /= 2**0.5	# to Voigt notation

			
			M = voigtVecToMat(ei_Tnf)
			ef = M - np.dot(np.dot(M, d), d)*P + P*ed
			ef = matToVoigtVec(ef)

			s += 0.5*l
			f.write("\t".join(map(str, [i, ei, s, iv1, iv2, *d, *um, *du, ed, *ef, *ei_nf, *ei_r, *ei_Tnf])) + "\n")
			s += 0.5*l
			i += 1

		f.close()

	return strain

	# Plot solution
	#plot(u)
	#plt.show()

# NOTE: the homogenization results have the fiber oriented in z direction, therefore we have to swap coordinates (swap_xz)
# to get the desired fiber orientation (i.e. x or y)

# matrix material
Cm = swap_coords_C(np.loadtxt("../homogenization/C_hom_%d.npy" % volfrac))
print("Cm = ")
print(Cm)

# fiber material
E_f = 73.0
nu_f = 0.18
lambda_f = nu_f/((1 - 2*nu_f)*(1 + nu_f))*E_f
mu_f = 0.5/(1 + nu_f)*E_f
Cf = 2*mu_f*np.eye(6)
Cf[0:3,0:3] += lambda_f
Cf[3,3] *= 0.5
Cf[4,4] *= 0.5
Cf[5,5] *= 0.5
print("Cf = ")
print(Cf)

# coating material
E_c = 2.5
nu_c = 0.34
lambda_c = nu_c/((1 - 2*nu_c)*(1 + nu_c))*E_c
mu_c = 0.5/(1 + nu_c)*E_c
Cc = 2*mu_c*np.eye(6)
Cc[0:3,0:3] += lambda_c
Cc[3,3] *= 0.5
Cc[4,4] *= 0.5
Cc[5,5] *= 0.5
print("Cc = ")
print(Cc)

if not coating_subdomain_id is None:
	T1_file += "_coated"
	T2_file += "_coated"

# strain transfer matrix (computed with ../homogenization/run_stp.py)
# (Ef = T*Em): T =
T1 = swap_coords_T1(np.loadtxt("../homogenization/T_%d_%s.npy" % (volfrac, T1_file)))
T2 = swap_coords_T2(np.loadtxt("../homogenization/T_%d_%s.npy" % (volfrac, T2_file)))
# inverse strain transfer matrix (Em = inv(T)*Ef): inv(T) =
print("T1_%s = " % T1_file)
print(T1)
print("T2_%s = " % T2_file)
print(T2)

if T_path is None:
	T_path = np.zeros((len(fiber_angle), 6, 6))
	for i in range(len(fiber_angle)):
		T_path[i] = np.cos(fiber_angle[i])**2 * T1 + np.sin(fiber_angle[i])**2 * T2
else:
	pass

invT_path = np.zeros_like(T_path)
for i in range(len(sample_path)):
	invT_path[i] = np.linalg.inv(T_path[i])

fiber_edges = None
if not fiber_path_id is None:
	fiber_edges = np.where(mesh_paths.array() == fiber_path_id)[0]
	# compute center of each edge and sort
	fiber_edge_center = {}
	ref_point = np.array(sample_path[0])
	for ei in fiber_edges:
		edge = Edge(mesh, ei)
		v1, v2 = vertices(edge)
		p1 = v1.point().array()
		p2 = v2.point().array()
		fiber_edge_center[ei] = 0.5*(p1+p2)
	def fiber_edge_distance(e):
		return np.linalg.norm(fiber_edge_center[e] - ref_point)
	fiber_edges = sorted(fiber_edges, key=fiber_edge_distance)

# solve with and without embedded fiber
# theoretically et11 of the solution with fiber should be e11 of the solution without fiber
# and et11 of the solution without fiber should be e11 of the solution with fiber
# (euqivalently for the other components)
extra = ""
postfix = "%d_%s%s%s%s" % (volfrac, reinforcement_dir, fiber_dir1, fiber_dir2, extra)

strain_nf = solve_stuff("nf_%s" % postfix, Cm, Cm, Cm, T_path, sample_path, None, None, None, r_f, E_f)	# without fiber
strain_r = solve_stuff("r_%s" % postfix, Cm, Cf, Cc, invT_path, sample_path, None, None, None, r_f, E_f)	# 3d fiber referenece solution
strain_f = solve_stuff("f_%s" % postfix, Cm, Cm, Cm, T_path, sample_path, strain_nf, strain_r, fiber_edges, r_f, E_f)	# with superimposed 1d fiber model

