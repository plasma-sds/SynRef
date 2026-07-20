from physics.basic import Basic
import numpy as np

sim = Basic(wavesource='pyramidal_horn', solver='antenna', frequency=40e9, angle=15, antenna_pos=0.2, E1=10, horn_x=-0.5, horn_a1=39.97e-3, horn_b1=30.588e-3, horn_rho1=30e-3, horn_rho2=60e-3) #'pyramidal_horn' gaussian
sim.run()
ez = sim.get_fields(kind='electric')
print("ez_final min/max:", ez.min(), ez.max())
sim.plot_results()