from reflectometer.basic import Basic
import numpy as np

sim = Basic(solver='antenna', frequency=29e9, antenna='W7X', angle=20, antenna_pos=0.2, ne_target=1e19)

run= True
if run:
    sim.run()
    ez = sim.get_fields(kind='electric')
    print("ez_final min/max:", ez.min(), ez.max())
    sim.plot_results()