"""Closed-loop sign following with bounded active re-observation.

Exact simulated pose, known observation stations and fixed route geometry are
assumptions. The classifier/policy receives only pixels. Ground truth is consulted
once, after a mission has ended, solely by the evaluator.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
from drive import DifferentialDrive,waypoint_command,alignment_command
from world import PracticeWorld,JUNCTION_Y,GOAL
from policy import DecisionPolicy,GateConfig,POLICIES
from vision import ROOT,SignClassifier

def run_episode(seed:int=63000,mode:str='active_recheck',condition:str='variable',save_dir=None,
                model=None,config=None,record=False):
    config=config or GateConfig()
    world=PracticeWorld(seed,condition)
    robot=DifferentialDrive(world.start,world.wheel_gain)
    if model is None:
        model=SignClassifier(ROOT/'models'/('baseline_mlp.npz' if mode=='baseline_once' else 'robust_mlp.npz'))
    agent=DecisionPolicy(mode,model,config)
    out=Path(save_dir) if save_dir is not None else None
    if out:out.mkdir(exist_ok=True,parents=True);(out/'camera').mkdir(exist_ok=True)
    dt=.01;control_every=5
    t=0.;step=0;stage='APPROACH';junction=0;view=0;path_index=0
    path=[world.stations(0)[0]];linear=angular=last_v=last_w=0.
    read_start=0.;decision=None;choices=[];events=[];observations=[];telemetry=[]
    reached_goal=False;held=False;contacts=0;rechecks=0;stopped_at=None;active_observation=-1
    def event(name,**extras):events.append(dict(time=round(t,3),event=name,junction=junction,**extras))
    event('START')
    while t<240.:
        if step%control_every==0:
            if stage in ('APPROACH','REPOSITION','FOLLOW_BRANCH','GOAL'):
                linear,angular,arrived=waypoint_command(robot.x,robot.y,robot.theta,path[path_index])
                if arrived:
                    path_index+=1
                    if path_index<len(path):
                        linear,angular,_=waypoint_command(robot.x,robot.y,robot.theta,path[path_index])
                    elif stage in ('APPROACH','REPOSITION'):
                        stage='ALIGN';linear=angular=0.;event('AT_OBSERVATION_STATION',view=view)
                    elif stage=='FOLLOW_BRANCH':
                        junction+=1;path_index=0;view=0;agent.reset()
                        if junction<3:stage='APPROACH';path=[world.stations(junction)[0]]
                        else:stage='GOAL';path=[GOAL]
                    else:
                        stage='FINISH';linear=angular=0.;event('GOAL_REACHED')
            elif stage=='ALIGN':
                linear,angular,aligned=alignment_command(robot.theta)
                if aligned and abs(last_v)<.008 and abs(last_w)<.08:
                    linear=angular=0.;stage='READ';read_start=t
                    image,actual_view=world.capture_sign(junction,(robot.x,robot.y,robot.theta))
                    assert actual_view==view
                    decision=agent.observe(image)
                    active_observation=len(observations)
                    fname=f'camera/j{junction+1}_v{view+1}.png'
                    if out:image.save(out/fname)
                    observations.append(dict(index=active_observation,junction=junction,view=view,
                        time=round(t,3),image=fname,image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        robot_pose=[robot.x,robot.y,robot.theta],**decision))
                    event('OBSERVE',view=view,action=decision['action'],prediction=decision['prediction'])
            elif stage=='READ':
                linear=angular=0.
                if t-read_start>=1.0:
                    if decision['action']=='GO':
                        choices.append(decision['choice']);stage='FOLLOW_BRANCH'
                        path=world.branch_points(junction,choices[-1]);path_index=0
                        event('COMMIT_BRANCH',choice=choices[-1])
                    elif decision['action']=='RECHECK':
                        view+=1;rechecks+=1;stage='REPOSITION';path_index=0
                        path=[world.stations(junction)[view]];event('RECHECK_MOVE',view=view)
                    else:
                        stage='HOLD';stopped_at=junction;event('HOLD_REQUEST')
            elif stage in ('FINISH','HOLD'):
                linear=angular=0.
                if abs(last_v)<.001 and abs(last_w)<.01:
                    held=stage=='HOLD'
                    reached_goal=stage=='FINISH' and math.hypot(robot.x-GOAL[0],robot.y-GOAL[1])<.08
                    event('HELD' if held else 'COMPLETE')
                    break
            if record or out:
                telemetry.append(dict(time=round(t,4),x=robot.x,y=robot.y,theta=robot.theta,
                    wheel_left=robot.left,wheel_right=robot.right,linear=last_v,angular=last_w,
                    command_linear=linear,command_angular=angular,state=stage,
                    junction=min(junction,2),view=view,observation_index=active_observation,
                    choice=choices[-1] if choices else -1))
        last_v,last_w=robot.step(linear,angular,dt)
        if world.collides(robot.x,robot.y):contacts+=1;event('CONTACT');break
        t+=dt;step+=1
    if not (reached_goal or held or contacts):event('TIME_LIMIT')
    ev=world.evaluate(choices,stopped_at)
    result=dict(seed=seed,mode=mode,condition=condition,completed=reached_goal,held=held,
        stopped_at=stopped_at,contacts=contacts,duration_seconds=round(t,3),
        path_length_m=robot.path_length,choices=choices,observations=len(observations),rechecks=rechecks,
        final_position=[robot.x,robot.y],max_wheel_acceleration=robot.max_observed_acceleration,
        **ev)
    result['route_success']=bool(reached_goal and ev['all_signs_followed'] and contacts==0)
    result['scenario_success']=bool((result['route_success'] if condition!='covered' else ev['correct_hold']) and contacts==0)
    result['unsafe_actions']=int(ev['wrong_decisions']+ev['covered_entries'])
    mission=dict(schema_version=2,result=result,gate=config.__dict__,events=events,observations=observations,
        simulation=dict(model='no-slip planar differential drive with wheel lag and acceleration limit',
            integration_step_seconds=dt,control_step_seconds=.05,
            sensor='synthetic sign patches at known observation positions; not a perspective camera',
            visibility='variable: stress at view 0, standard at views 1/2; covered: panel 2 blank at every view; persistent: stress at all views',
            localization='exact simulator pose',collision='circle-rectangle overlap; stop on contact',
            wheel_radius_m=robot.radius,axle_m=robot.axle,wheel_tau_seconds=robot.tau,
            wheel_acceleration_limit_rad_s2=robot.acceleration_limit,wheel_gain=list(world.wheel_gain)))
    if out:
        (out/'mission.json').write_text(json.dumps(mission,indent=2))
        (out/'world.json').write_text(json.dumps(world.geometry(),indent=2))
        if telemetry:
            with (out/'telemetry.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(telemetry[0]));w.writeheader();w.writerows(telemetry)
    return dict(result=result,mission=mission,telemetry=telemetry,world=world.geometry())

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=63000)
    p.add_argument('--mode',choices=POLICIES,default='active_recheck')
    p.add_argument('--condition',choices=['normal','variable','covered','persistent'],default='variable')
    p.add_argument('--out',type=Path,default=ROOT/'results/demo_active')
    a=p.parse_args();print(json.dumps(run_episode(a.seed,a.mode,a.condition,a.out)['result'],indent=2))
if __name__=='__main__':main()
