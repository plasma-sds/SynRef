from reflectometer.basic import Basic
import numpy as np

sim = Basic(solver='antenna', frequency=40e9, antenna='W7X', angle=-15, antenna_pos=0.2)

run= False
if run:
    sim.run()
    ez = sim.get_fields(kind='electric')
    print("ez_final min/max:", ez.min(), ez.max())
    sim.plot_results()