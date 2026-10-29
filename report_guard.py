"""Fail closed if fixed explanatory captions no longer match the recorded facts.

The supplied report is an authored explanation of one measured experiment, not a
universal templated analysis. A modified experiment needs revised explanatory
text. This check prevents stale claims from being silently re-exported.
"""
from __future__ import annotations
import json
from vision import ROOT

def current_facts():
    b=json.loads((ROOT/'results/benchmark/summary.json').read_text())
    v=json.loads((ROOT/'results/vision_metrics.json').read_text())
    a=json.loads((ROOT/'results/verification.json').read_text())
    selection=json.loads((ROOT/'results/demo_selection.json').read_text())
    groups=[]
    for g in b['groups']:
        groups.append({k:round(val,8) if isinstance(val,float) else val for k,val in g.items()})
    return dict(groups=groups,images=v['evaluations'],demo_selection=selection,
                observations_reinferred=a['observations_reinferred'],missions_audited=a['missions_audited'],
                default_gate=b['protocol']['gate'],demo_results={name:json.loads((ROOT/'results'/name/'mission.json').read_text())['result']
                  for name in ('demo_baseline','demo_once','demo_active','demo_hold')})

def assert_report_snapshot():
    saved=json.loads((ROOT/'results/report_snapshot.json').read_text())
    if current_facts()!=saved:
        raise RuntimeError('Recorded facts changed. Revise the PDF/video explanatory text and review results before regenerating media; do not reuse the old performance claims.')
