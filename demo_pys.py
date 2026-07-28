from reflectometer.basic import Basic
import numpy as np

sim = Basic(solver='antenna', frequency=20e9, antenna='default', angle=0, antenna_pos=0.2)
sim.run()
ez = sim.get_fields(kind='electric')
print("ez_final min/max:", ez.min(), ez.max())
sim.plot_results()