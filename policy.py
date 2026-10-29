"""Pixel-only decisions. No target labels, scenario IDs or environment object enter here."""
from __future__ import annotations
from dataclasses import dataclass
from PIL import Image
from vision import SignClassifier, readability
POLICIES=('baseline_once','robust_once','robust_gate','active_recheck')

@dataclass(frozen=True)
class GateConfig:
    threshold: float = .95
    contrast_min: float = 25.
    max_views: int = 3
    required_agreement: int = 2

class DecisionPolicy:
    def __init__(self,mode:str,model:SignClassifier,config:GateConfig):
        if mode not in POLICIES: raise ValueError('unknown policy')
        self.mode,self.model,self.config=mode,model,config
        self.history=[]
    def reset(self):self.history=[]
    def observe(self,image:Image.Image):
        prediction=self.model.predict(image)
        contrast=readability(image)
        reliable=bool(contrast>=self.config.contrast_min and prediction['confidence']>=self.config.threshold)
        item=dict(prediction=prediction,contrast=contrast,reliable=reliable)
        self.history.append(item)
        n=len(self.history)
        decision='RECHECK'
        if self.mode.endswith('_once'):
            decision='GO'
        elif self.mode=='robust_gate':
            decision='GO' if reliable else 'HOLD'
        else:
            recent=self.history[-self.config.required_agreement:]
            consistent=(len(recent)==self.config.required_agreement and
                        all(v['reliable'] for v in recent) and
                        len({v['prediction']['label'] for v in recent})==1)
            if consistent:decision='GO'
            elif n>=self.config.max_views:decision='HOLD'
        return {**item,'action':decision,'view_count':n,
                'choice':int(prediction['label']) if decision=='GO' else None}
