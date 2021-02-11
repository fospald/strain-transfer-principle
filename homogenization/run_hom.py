
import fibergen
import numpy as np

# create solver instance
fg = fibergen.FG()

fibervolfrac_maxpacking = np.pi*3**0.5/6.0
fibervolfrac = 0.5

# load project XML
# alternative: fg.load_xml("project.xml")
fg.load_xml("project_hom.xml")
fg.set("variables.R..value", 0.5*(fibervolfrac/fibervolfrac_maxpacking)**0.5)

# run solver
fg.run()

C = fg.get_effective_property()

print("C =", C)

np.savetxt("C_hom_%d.npy" % np.round(fibervolfrac*100), C)

