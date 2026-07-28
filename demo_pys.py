from reflectometer.basic import Basic
import numpy as np

sim = Basic(wavesource='pyramidal_horn', solver='antenna', frequency=20e9, angle=0, antenna_pos=0.2, horn='W7X') #'pyramidal_horn' gaussian
sim.run()
ez = sim.get_fields(kind='electric')
print("ez_final min/max:", ez.min(), ez.max())
sim.plot_results()