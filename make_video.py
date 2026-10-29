"""Create a 1-3 minute, 1080p captioned video from saved simulation logs."""
from __future__ import annotations
import argparse,json,math,shutil,subprocess
from pathlib import Path
from video_ui import ROOT,W,H,Scene,paired,hold_screen,intro,detail,summary_screen

def main():
    from report_guard import assert_report_snapshot
    assert_report_snapshot()
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'deliverables/動画_PAI最終課題_Toshigakusei.mp4');p.add_argument('--preview',action='store_true')
    a=p.parse_args();fps=20;baseline=Scene('demo_baseline');active=Scene('demo_active');hold=Scene('demo_hold')
    out=ROOT/'results/figures';out.mkdir(exist_ok=True,parents=True)
    stills={'intro':intro(),'comparison':paired(baseline,active,66),
            'observations':detail(active),'hold':hold_screen(hold,hold.duration),'summary':summary_screen()}
    for name,im in stills.items():im.save(out/(name+'.png'))
    if a.preview:return
    if not shutil.which('ffmpeg'):raise RuntimeError('FFmpeg is required to render video.')
    a.out.parent.mkdir(exist_ok=True,parents=True)
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','-',
         '-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);n=0;sections=[]
    def static(name,seconds):
        nonlocal n
        b=stills[name].tobytes();frames=int(seconds*fps)
        for _ in range(frames):proc.stdin.write(b)
        n+=frames;sections.append(dict(name=name,seconds=seconds,playback='still caption'))
    def movie(name,duration,fn):
        nonlocal n
        frames=math.ceil(duration/2*fps)+1
        for i in range(frames):
            proc.stdin.write(fn(min(i/fps*2,duration)).tobytes())
        n+=frames;sections.append(dict(name=name,seconds=frames/fps,simulation_seconds=duration,playback=2))
    try:
        static('intro',8)
        movie('matched_variable',max(baseline.duration,active.duration),lambda t:paired(baseline,active,t))
        print('paired video rendered',flush=True)
        static('observations',10)
        movie('covered_panel',hold.duration,lambda t:hold_screen(hold,t))
        print('hold video rendered',flush=True)
        static('hold',4)
        static('summary',16)
    finally:proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('FFmpeg encoding failed')
    meta=dict(width=W,height=H,fps=fps,frames=n,duration_seconds=n/fps,sections=sections,
              audio='none; Japanese explanatory captions',source='saved wheel-kinematics telemetry; no scripted robot animation')
    (ROOT/'results/video_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
