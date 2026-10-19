"""Validation of horn_near_field.py against the CST 29 GHz exports (z=52 mm -> z=158 mm).
Usage: python validate_horn_near_field.py <folder containing output_E_52.txt etc.>
"""
import numpy as np, glob, time, json
from horn_near_field import *
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]       
EXPORT_DIR = REPO_ROOT / "hardware/utils/antenna/field_exports/W7X_PCR"

ap=load_aperture(str(EXPORT_DIR / "output_E_52.txt"),29e9,h_file=str(EXPORT_DIR / "output_H_52.txt"))
rE=load_field_txt(str(EXPORT_DIR / "output_E_158.txt")); rH=load_field_txt(str(EXPORT_DIR / "output_H_158.txt"))  #reference CST field at z=158 mm
d=rE.z-ap.center[2]; #distance from aperture to CST reference plane 

out={} 

print("P52 = %.4f W"%ap.power_z())

for L in (0.04,0.05):

  m=(np.abs(rE.x)[None,:]<=L)&(np.abs(rE.y)[:,None]<=L)

  for w in ("total","forward"):
    t=time.time(); E,H=propagate_parallel(ap,d,wave=w); dt=time.time()-t

    eE, a = nrmse(E[:,m],rE.F[:,m])
    eH, _ = nrmse(H[:,m],rH.F[:,m])

    print(f"parallel |x|,|y|<={L*1e3:.0f}mm wave={w:8s} NRMSE E-Field {eE:.4f} NRMSE H-Field {eH:.4f} |a|={abs(a):.4f} ang={np.angle(a,deg=True):.2f}  comps",
          [round(nrmse(E[c][m],rE.F[c][m])[0],4) for c in range(3)], f"{dt:.1f}s")
    
# direct-sum untilted on coarse grid
xq=rE.x[::4]; 
yq=rE.y[::4]; 
ms=((np.abs(xq)[None,:]<=0.04)&(np.abs(yq)[:,None]<=0.04))

t=time.time(); 
E,H,info = propagate_to_z_plane(ap,rE.z,xq,yq); 
print("direct untilted", info, "%.1fs"%(time.time()-t), "NRMSE E-Field %.4f"%nrmse(E[:,ms],rE.F[:,::4,::4][:,ms])[0])
SC0,_,_= propagate_to_z_plane(ap,rE.z,xq,yq,method='sc',src_step=2); 
print("direct vs SC(untilted) %.4f"%nrmse(E[:,ms],SC0[:,ms],False)[0])

# tilted
xq=np.linspace(-0.05,0.05,41)
poses={"ry=+10":dict(ry_deg=10),"ry=-10":dict(ry_deg=-10),"ry=20":dict(ry_deg=20),"rx=15,rz=30":dict(rx_deg=15,rz_deg=30),
       "ry=10 about throat":dict(ry_deg=10,pivot=(0,0,0.0))}
res={}
for name,kw in poses.items():
  pose=AntennaPose.from_tilt(ap.center,**kw)
  t=time.time(); Es,Hs,i1=propagate_to_z_plane(ap,rE.z,xq,xq,pose); t1=time.time()-t
  t=time.time(); Ec,Hc,i2=propagate_to_z_plane(ap,rE.z,xq,xq,pose,method="stratton_chu",src_step=2); t2=time.time()-t
  res[name]=Es
  hit=pose.boresight_hit_z(rE.z)*1e3
  print(f"{name:20s} spectrum({i1['pad']},{i1['modes']} modes,{t1:.1f}s) vs SC({t2:.1f}s): E field NRMSE {nrmse(Es,Ec,False)[0]:.4f} H field NRMSE {nrmse(Hs,Hc,False)[0]:.4f}  z' range {np.round(np.array(i1['z_antenna_range'])*1e3,1)} mm  boresight hit {np.round(hit,1)} mm")

# mirror symmetry check
Ep,Em=res["ry=+10"],res["ry=-10"]
print("mirror |E| ry=+10 vs ry=-10 (x->-x):", np.linalg.norm(np.abs(Ep)-np.abs(Em[:,:,::-1]))/np.linalg.norm(np.abs(Ep)))
# SC src_step accuracy

pose=AntennaPose.from_tilt(ap.center,ry_deg=20); xs=np.linspace(-0.05,0.05,15)
E1,_,_=propagate_to_z_plane(ap,rE.z,xs,xs,pose,method="sc",src_step=1)
E2,_,_=propagate_to_z_plane(ap,rE.z,xs,xs,pose,method="sc",src_step=2)
print("SC step2 vs step1: %.5f"%nrmse(E2,E1,False)[0])
