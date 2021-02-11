
# title: Pure Python
# description: Shows how to use fibergen in Python.

import fibergen
import numpy as np
import sys

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

def calc_stuff(coated, volfrac, swap_axes, postfix=""):

	# create solver instance
	fg = fibergen.FG()

	# load project XML
	# alternative: fg.load_xml("project.xml")
	if coated:
		fg.load_xml("project_stp_coated.xml")
		postfix += "_coated"
	else:
		fg.load_xml("project_stp.xml")

	if not volfrac is None:
		# homogenized matrix material from run_hom.py
		C = np.loadtxt("C_hom_%d.npy" % volfrac)
		C = swap_axes(C)

		for i in range(6):
			for j in range(6):
				fg.set('solver.materials.matrix..c%d%d' % (i+1, j+1), C[i][j])

	#print(fg.get_xml())
	#sys.exit(0)

	# identify strain transfer matrix
	Em = np.eye(6) # applied matrix strains
	Ef = np.eye(6) # resulting strain inside the FBG
	for i in range(6):
		Em_i = Em[:,i]
		fg.set('actions.run_load_case.', e11=Em_i[0], e22=Em_i[1], e33=Em_i[2], e23=Em_i[3], e13=Em_i[4], e12=Em_i[5])
		# run solver
		fg.run()
		# get strain inside the FBG
		eps = fg.get_field("epsilon")
		Ef[:,i] = eps[:,int(eps.shape[1]/2), int(eps.shape[2]/2), 0]

	# compute strain transfer matrix
	# Ef = T*Em <=> T = Ef*inv(Em)
	# Mandel notation
	Ef[3:6,:] *= 2**0.5
	Em[3:6,:] *= 2**0.5
	T = np.matmul(Ef, np.linalg.inv(Em))
	np.set_printoptions(precision=4, suppress=True)
	print("strain transfer matrix (Ef = T%s*Em): T%s =" % (postfix, postfix))
	print(T)
	print("inverse strain transfer matrix (Em = inv(T%s)*Ef): inv(T%s) =" % (postfix, postfix))
	print(np.linalg.inv(T))

	if not volfrac is None:
		np.savetxt("T_%d%s.npy" % (volfrac, postfix), T)


coated = False
volfrac = 50

calc_stuff(coated, volfrac, swap_none, "_zz")
calc_stuff(coated, volfrac, swap_xz, "_xz")

for i in range(91):
	alpha = i/90.0*np.pi/2.0
	swap_alpha = lambda C: tensorToVoigtMatrix(rotateTensor(voigtMatrixToTensor(C), rotateX(alpha)))
	calc_stuff(coated, volfrac, swap_alpha, "_zx%d" % i)

