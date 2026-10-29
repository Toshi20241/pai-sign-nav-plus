import math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from PIL import Image,ImageDraw
from policy import DecisionPolicy,GateConfig
from vision import CLASS_NAMES,SignClassifier,render_sign,readability,preprocess,ROOT
from world import PracticeWorld,CONDITIONS
from drive import DifferentialDrive,waypoint_command

class Stub:
    def __init__(self,labels,scores=None):self.labels=iter(labels);self.scores=iter(scores or [.99]*len(labels))
    def predict(self,image):
        label=next(self.labels);s=next(self.scores);p=[(1-s)/2]*3;p[label]=s
        return dict(label=label,name=CLASS_NAMES[label],confidence=s,probabilities=p)

def valid_image():return render_sign(0,np.random.default_rng(5),'canonical')
def decision(labels,scores=None):return DecisionPolicy('active_recheck',Stub(labels,scores),GateConfig())

def test_first_view_never_accepted():assert decision([0]).observe(valid_image())['action']=='RECHECK'
def test_agreement():
    p=decision([1,1]);assert p.observe(valid_image())['action']=='RECHECK';assert p.observe(valid_image())['action']=='GO'
def test_disagreement_retries():
    p=decision([0,1]);p.observe(valid_image());assert p.observe(valid_image())['action']=='RECHECK'
def test_third_disagreement_holds():
    p=decision([0,1,2]);p.observe(valid_image());p.observe(valid_image());assert p.observe(valid_image())['action']=='HOLD'
def test_third_consensus_accepts():
    p=decision([0,1,1]);p.observe(valid_image());p.observe(valid_image());assert p.observe(valid_image())['choice']==1
def test_low_scores_hold():
    p=decision([0,0,0],[.5,.5,.5]);p.observe(valid_image());p.observe(valid_image());assert p.observe(valid_image())['action']=='HOLD'
def test_covered_high_score_holds():
    p=decision([0,0,0]);im=Image.new('RGB',(96,96),(220,220,220));p.observe(im);p.observe(im);assert p.observe(im)['action']=='HOLD'
def test_reset():
    p=decision([0,0]);p.observe(valid_image());p.reset();assert p.observe(valid_image())['action']=='RECHECK'
def test_single_gate():
    p=DecisionPolicy('robust_gate',Stub([0],[.5]),GateConfig());assert p.observe(valid_image())['action']=='HOLD'
def test_once():
    p=DecisionPolicy('robust_once',Stub([2],[.4]),GateConfig());assert p.observe(valid_image())['choice']==2
def test_bad_mode():
    with pytest.raises(ValueError):DecisionPolicy('bad',Stub([0]),GateConfig())
def test_features():assert preprocess(valid_image()).shape==(576,)
def test_class_probabilities():
    p=SignClassifier().predict(valid_image())['probabilities'];assert abs(sum(p)-1)<1e-6;assert len(p)==3
def test_blank_contrast():assert readability(Image.new('RGB',(96,96),'white'))==0
def test_valid_contrast():assert readability(valid_image())>25
@pytest.mark.parametrize('condition',CONDITIONS)
def test_sensor_determinism(condition):
    a=PracticeWorld(63001,condition);b=PracticeWorld(63001,condition)
    x,y=a.stations(0)[0];im,view=a.capture_sign(0,(x,y,math.pi/2));im2,v2=b.capture_sign(0,(x,y,math.pi/2))
    assert im.tobytes()==im2.tobytes() and view==v2==0
def test_capture_pose_guard():
    with pytest.raises(ValueError):PracticeWorld(1).capture_sign(0,(0,8,0))
def test_geometry_no_labels():assert 'true_labels' not in PracticeWorld(1).geometry()
def test_covered_visibility():
    w=PracticeWorld(1,'covered')
    assert all(readability(w._panel(1,v))<25 for v in range(3))
def test_wheel_acceleration():
    r=DifferentialDrive((0,0,math.pi/2))
    for _ in range(20):r.step(1,1,.01)
    assert r.max_observed_acceleration<=32
    assert r.y>0
def test_geometry_collision():
    w=PracticeWorld(1);assert w.collides(-.65,2.65);assert not w.collides(0,0)
def test_no_goal_credit_for_hold():
    e=PracticeWorld(1,'covered').evaluate([],0);assert not e['all_signs_followed'];assert not e['correct_hold']
def test_branch_routes_distinct():
    a=PracticeWorld.branch_points(0,0);b=PracticeWorld.branch_points(0,2);assert a[1][0]<0<b[1][0]
def test_waypoint_stop():
    v,w,done=waypoint_command(0,0,0,(0,0));assert done and v==w==0
