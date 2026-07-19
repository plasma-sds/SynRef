from physics.basic import Basic
import numpy as np

sim = Basic(wavesource='gaussian', solver='antenna', frequency=40e9, angle=15, antenna_pos=0.15)
sim.run()
ez = sim.get_fields(kind='electric')
print("ez_final min/max:", ez.min(), ez.max())
sim.plot_results()