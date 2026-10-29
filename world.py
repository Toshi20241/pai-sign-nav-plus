"""Synthetic three-junction robot puzzle; private labels belong to evaluation only.

The sensor is a procedurally generated *sign patch*, not a rendered optical scene.
At three known observation stations its visibility improves in the VARIABLE case.
The COVERED case permanently hides the second arrow at all three positions.
Images are fixed by seed, junction and viewpoint, identical for all policies.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from vision import render_sign

JUNCTION_Y = (1.2,4.4,7.6)
GOAL = (0.,10.85)
ROBOT_RADIUS = .18
CONDITIONS = ('normal','variable','covered','persistent')

@dataclass(frozen=True)
class Rectangle:
    x: float
    y: float
    width: float
    height: float
    kind: str = 'pillar'
    def as_dict(self):
        return dict(x=self.x,y=self.y,width=self.width,height=self.height,kind=self.kind)

class PracticeWorld:
    def __init__(self,seed:int,condition:str='normal'):
        if condition not in CONDITIONS: raise ValueError('unknown condition')
        self.seed,self.condition=int(seed),condition
        r=np.random.default_rng(seed)
        self._labels=r.integers(0,3,3).astype(int).tolist()
        self.start=(float(r.uniform(-.04,.04)),-.35,math.pi/2+float(r.uniform(-.04,.04)))
        self.wheel_gain=tuple(map(float,r.uniform(.98,1.02,2)))
        self.obstacles=[Rectangle(x,y+1.45,.34,.90) for y in JUNCTION_Y for x in (-.65,.65)]
        self.obstacles += [Rectangle(-2.05,5.25,.12,12.3,'wall'),Rectangle(2.05,5.25,.12,12.3,'wall'),
                           Rectangle(0,-.88,4.22,.12,'wall'),Rectangle(0,11.38,4.22,.12,'wall')]
        self._bounds=[(o.x-o.width/2,o.x+o.width/2,o.y-o.height/2,o.y+o.height/2) for o in self.obstacles]
    @staticmethod
    def stations(junction:int):
        y=JUNCTION_Y[junction]
        # Before the first obstacles: bounded observation movements, no branch entry.
        return [(0.,y-.35),(-.23,y-.13),(.23,y-.04)]
    @staticmethod
    def branch_points(junction:int,label:int):
        y=JUNCTION_Y[junction]
        if label==1:return [(0.,y+.32),(0.,y+2.75)]
        x=(-1 if label==0 else 1)*1.35
        return [(0.,y+.05),(x,y+.65),(x,y+2.25),(0.,y+2.75)]
    def _panel(self,junction:int,view:int):
        rng=np.random.default_rng(self.seed*101+900_001+junction*11+view)
        if self.condition=='covered' and junction==1:
            bg=int(rng.integers(180,240))
            im=Image.new('RGB',(96,96),(bg,bg,bg))
            d=ImageDraw.Draw(im);d.rounded_rectangle((5,5,91,91),9,outline=(bg-15,)*3,width=1)
            noise=rng.normal(0,2.0,(96,96,1))
            return Image.fromarray(np.uint8(np.clip(np.array(im).astype(float)+noise,0,255)))
        profile='stress' if self.condition=='persistent' else ('standard' if self.condition=='normal' or view>=1 else 'stress')
        return render_sign(self._labels[junction],rng,profile)
    def capture_sign(self,junction:int,pose):
        x,y,theta=pose
        distances=[math.hypot(x-a,y-b) for a,b in self.stations(junction)]
        view=int(np.argmin(distances))
        yaw=(theta-math.pi/2+math.pi)%(2*math.pi)-math.pi
        if distances[view]>.065 or abs(yaw)>.06:
            raise ValueError('capture requires a reached observation station and aligned heading')
        return self._panel(junction,view),view
    def collides(self,x,y):
        r2=ROBOT_RADIUS**2
        for a,b,c,d in self._bounds:
            dx=x-min(b,max(a,x)); dy=y-min(d,max(c,y))
            if dx*dx+dy*dy<r2:return True
        return False
    def evaluate(self,choices,stopped_at):
        valid=[i for i in range(len(choices)) if not (self.condition=='covered' and i==1)]
        errors=sum(choices[i]!=self._labels[i] for i in valid)
        covered_entries=int(self.condition=='covered' and len(choices)>1)
        return dict(true_labels=self._labels.copy(),wrong_decisions=errors,
                    evaluated_decisions=len(valid),covered_entries=covered_entries,
                    correct_hold=bool(self.condition=='covered' and stopped_at==1 and errors==0),
                    all_signs_followed=bool(len(choices)==3 and errors==0 and covered_entries==0))
    def geometry(self):
        return dict(bounds=[-2,2,-.82,11.32],start=list(self.start),goal=list(GOAL),
                    junction_y=list(JUNCTION_Y),robot_radius=ROBOT_RADIUS,
                    obstacles=[o.as_dict() for o in self.obstacles],
                    stations=[self.stations(i) for i in range(3)],
                    branches=[{k:self.branch_points(i,l) for l,k in enumerate(('left','straight','right'))} for i in range(3)])
