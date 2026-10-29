"""Visualize recorded trajectories, not hand-authored robot motion.

Fonts are resolved from the local OS or PAI_FONT_REGULAR / PAI_FONT_BOLD.
No font binaries are bundled. Japanese rendering requires an installed CJK font.
"""
from __future__ import annotations
import csv,json,math,os
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from vision import ROOT,CLASS_NAMES
W,H=1920,1080
BG='#101B2C';PANEL='#18283D';EDGE='#34475E';WHITE='#F4F6F9';MUTED='#B8C5D3';AQUA='#5CE1CD';GOLD='#FFD27A';RED='#FF9F94'

@lru_cache(maxsize=80)
def font(size,bold=False):
    env=os.environ.get('PAI_FONT_BOLD' if bold else 'PAI_FONT_REGULAR')
    suffix='Bold' if bold else 'Regular'
    candidates=[env,f'/usr/share/fonts/opentype/noto/NotoSansCJK-{suffix}.ttc',
                '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc','C:/Windows/Fonts/meiryo.ttc']
    for path in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(path,int(size))
    raise RuntimeError('Install a Japanese CJK font or set PAI_FONT_REGULAR and PAI_FONT_BOLD to local font paths.')

def text(d,xy,s,size=26,fill=WHITE,bold=False,anchor=None):d.text(xy,str(s),font=font(size,bold),fill=fill,anchor=anchor)
def card(d,box,fill=PANEL):d.rounded_rectangle(box,radius=18,fill=fill,outline=EDGE,width=2)
def header(subtitle,right='Toshigakusei | 合成標識・平面車輪モデル'):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    text(d,(54,26),'PHYSICAL AI / FINAL PROJECT',18,AQUA,True)
    text(d,(50,57),'SIGN NAV +',62,WHITE,True)
    text(d,(505,80),subtitle,34,WHITE,True)
    text(d,(1865,30),right,19,MUTED,anchor='ra')
    d.line((50,147,1870,147),fill=EDGE,width=2)
    return im

class Scene:
    def __init__(self,sub):
        self.path=ROOT/'results'/sub
        self.world=json.loads((self.path/'world.json').read_text())
        self.mission=json.loads((self.path/'mission.json').read_text())
        self.result=self.mission['result'];self.duration=self.result['duration_seconds']
        with (self.path/'telemetry.csv').open() as f:self.rows=list(csv.DictReader(f))
        self.times=np.array([float(r['time']) for r in self.rows]);self.obs=self.mission['observations']
        self.images=[Image.open(self.path/o['image']).convert('RGB') for o in self.obs]
        self._bases={}
    def at(self,t):
        idx=max(0,min(len(self.rows)-1,int(np.searchsorted(self.times,t,side='right')-1)))
        return idx,self.rows[idx]
    def panel(self,im,t,box,title,accent=AQUA):
        left,top,width,height=box
        d=ImageDraw.Draw(im);card(d,(left,top,left+width,top+height))
        text(d,(left+22,top+16),title,28,WHITE,True)
        # Physical y progresses left-to-right; x is vertical on this view.
        sx=(width-76)/12.2;sy=52 if width<1000 else 64
        cy=top+220
        def proj(x,y):return (int(left+38+(y+.82)*sx),int(cy+x*sy))
        mapbox=(left+20,top+75,left+width-20,top+355)
        d.rounded_rectangle(mapbox,radius=12,fill='#21354D')
        for branch in self.world['branches']:
            for key in ('left','straight','right'):
                pp=branch[key];y=pp[0][1]-.05 if key!='straight' else pp[0][1]-.32
                pts=[(0,y-.35)]+pp
                d.line([proj(*p) for p in pts],fill='#657A90',width=4,joint='curve')
        d.line([proj(0,-.35),proj(0,10.85)],fill='#657A90',width=4)
        for ob in self.world['obstacles']:
            if ob['kind']=='pillar':
                a=proj(ob['x']-ob['width']/2,ob['y']-ob['height']/2)
                b=proj(ob['x']+ob['width']/2,ob['y']+ob['height']/2)
                d.rectangle((a[0],a[1],b[0],b[1]),fill='#40566F',outline='#93A3B5')
        for j,stations in enumerate(self.world['stations']):
            for vx,vy in stations:
                x,y=proj(vx,vy);d.ellipse((x-3,y-3,x+3,y+3),fill=GOLD)
            x,y=proj(0,stations[0][1]);text(d,(x,top+89),f'SIGN {j+1}',17,MUTED,True,anchor='ma')
        idx,row=self.at(min(t,self.duration))
        pts=[proj(float(r['x']),float(r['y'])) for r in self.rows[:idx+1]]
        if len(pts)>1:d.line(pts,fill=accent,width=5,joint='curve')
        x,y=proj(float(row['x']),float(row['y']));th=float(row['theta'])
        f=np.array([math.sin(th),math.cos(th)]);side=np.array([-f[1],f[0]])
        corners=[tuple(np.array([x,y])+f*a+side*b) for a,b in [(-15,-11),(15,-11),(15,11),(-15,11)]]
        d.polygon(corners,fill=GOLD,outline='#FFF0CD')
        for sign in (-1,1):
            a=np.array([x,y])+f*(-11)+side*(sign*15);b=np.array([x,y])+f*11+side*(sign*15)
            d.line((tuple(a),tuple(b)),fill='#0A101A',width=6)
        d.line(((x,y),tuple(np.array([x,y])+f*30)),fill=WHITE,width=3)
        gx,gy=proj(0,10.85);d.ellipse((gx-9,gy-9,gx+9,gy+9),outline=AQUA,width=3)
        text(d,(gx,gy+20),'GOAL',15,MUTED,anchor='ma')
        text(d,(left+25,top+365),'灰：既知経路  色：走行軌跡  黄点：観測位置',18,MUTED)
        latest=-1
        for i,o in enumerate(self.obs):
            if o['time']<=t+1e-6:latest=i
        y0=top+410
        if latest>=0:
            ob=self.obs[latest];image=self.images[latest];pred=ob['prediction']
            im.paste(image.resize((155,155)),(left+28,y0))
            values=pred['probabilities'];action=ob['action'];view=ob['view']+1
            label=['左','直進','右'][pred['label']]
            details=[f"SIGN {ob['junction']+1} / 観測 {view}",
                     f"判定：{label} / スコア {pred['confidence']:.3f}",
                     f"画像コントラスト：{ob['contrast']:.1f}"]
        else:
            d.rounded_rectangle((left+28,y0,left+183,y0+155),radius=8,fill='#32445B')
            values=[0,0,0];action='APPROACH';details=['最初の観測位置へ','まだ画像は入力していません','']
        for k,line in enumerate(details):text(d,(left+210,y0+k*41),line,23,WHITE if k<2 else MUTED,k==0)
        for k,(name,v) in enumerate(zip(['左','直進','右'],values)):
            xx=left+210+k*215;yy=y0+133
            text(d,(xx,yy),name,18,MUTED)
            d.rounded_rectangle((xx+47,yy+9,xx+185,yy+20),radius=4,fill='#354A63')
            if v>0:d.rounded_rectangle((xx+47,yy+9,xx+47+max(3,int(138*v)),yy+20),radius=4,fill=accent)
        phases={'APPROACH':'標識へ接近','REPOSITION':'別の観測位置へ移動','ALIGN':'停止して向きを調整',
                'READ':'画像を読み取り','FOLLOW_BRANCH':'確定した枝を走行','GOAL':'ゴールへ移動','FINISH':'ゴールで停止','HOLD':'判断保留で停止'}
        state=phases.get(row['state'],row['state'])
        if t>=self.duration:
            if self.result['held']:state='HOLD / 判断保留・停止'
            elif self.result['route_success']:state='完走 / 3標識の指示に一致'
            else:state='到着したが、標識と違う枝あり'
        yy=top+height-110
        d.line((left+22,yy-12,left+width-22,yy-12),fill=EDGE,width=2)
        text(d,(left+25,yy),state,27,GOLD if self.result['held'] and t>=self.duration else accent,True)
        text(d,(left+25,yy+47),f"左右輪 {float(row['wheel_left']):+.2f} / {float(row['wheel_right']):+.2f} rad/s",20,MUTED)
        text(d,(left+width-25,yy+47),f"t = {min(t,self.duration):05.1f} s",22,MUTED,anchor='ra')
        return im

def paired(a,b,t):
    im=header('一度だけ読む方式と、再読取りする方式')
    a.panel(im,t,(45,178,900,762),'旧モデル・一回判定',RED)
    b.panel(im,t,(975,178,900,762),'改良モデル・再読取り',AQUA)
    d=ImageDraw.Draw(im)
    text(d,(55,965),'同じ標識・初期位置・車輪条件で比較。位置を変えると画像の見え方が変化する設定。',26,MUTED)
    text(d,(55,1015),'記録済み走行ログを表示 / 走行 2倍速 / AIは画像分類、車輪は経由点へのフィードバック制御',21,MUTED)
    return im

def hold_screen(scene,t):
    im=header('見えなければ、進路を決めずに止まる')
    scene.panel(im,t,(45,178,1260,762),'全面被覆 / 同じ制御を変更せずに実行',AQUA)
    d=ImageDraw.Draw(im);card(d,(1330,178,1875,940))
    for y,title,line in [(220,'2番目の矢印を隠す','どの観測位置でも矢印は見えない'),
                         (420,'観測は最大3回','無限に読み直さず、回数を制限'),
                         (620,'判断保留で停止','停止を「完走」とは数えない')]:
        text(d,(1358,y),title,30,GOLD,True);text(d,(1358,y+66),line,24,MUTED)
    text(d,(55,975),'コントラストと出力スコアが条件を満たさず、分岐へ入る前に停止。',28,WHITE,True)
    text(d,(55,1020),'記録済み走行ログ / 走行 2倍速 / 全面被覆の検出は一般的な未知物体検出ではない',22,MUTED)
    return im

def intro():
    im=header('読み直して進む。読めなければ止まる。');d=ImageDraw.Draw(im)
    text(d,(80,225),'一枚の認識結果で、走り出してよいか？',55,WHITE,True)
    text(d,(85,325),'架空の矢印パネルを読む移動ロボットで、認識と行動のつながりを検証。',31,MUTED)
    for i,(a,b,c) in enumerate([('01  見る','小さな画像AI','左・直進・右を分類'),('02  読み直す','位置を変えて再観測','連続2回の一致を確認'),('03  保留する','最大3回で打ち切り','読めないまま進まない')]):
        x=80+i*605;card(d,(x,465,x+550,790));text(d,(x+28,495),a,38,AQUA,True)
        text(d,(x+28,597),b,31,WHITE,True);text(d,(x+28,664),c,27,MUTED)
    text(d,(85,875),'4方式 × 4条件 × 30試行 = 480走行 / 学習と評価は別の乱数系列',29,WHITE,True)
    text(d,(85,980),'合成画像・既知コース・自己位置既知の平面シミュレーション。実機・実カメラの評価ではありません。',23,MUTED)
    return im

def detail(scene):
    im=header('再読取りの根拠を、入力画像で確認');d=ImageDraw.Draw(im)
    seq=[(i,o) for i,o in enumerate(scene.obs) if o['junction']==0]
    for k,(i,o) in enumerate(seq):
        x=85+k*610;card(d,(x,215,x+550,820));text(d,(x+28,242),f'観測 {k+1}',32,AQUA,True)
        im.paste(scene.images[i].resize((270,270)),(x+140,310))
        text(d,(x+28,620),f"出力：{['左','直進','右'][o['prediction']['label']]}  /  {o['prediction']['confidence']:.3f}",29,WHITE,True)
        text(d,(x+28,686),f"コントラスト：{o['contrast']:.1f}",25,MUTED)
        text(d,(x+28,742),{'RECHECK':'再読取りへ','GO':'一致を確認 → 進む','HOLD':'判断保留'}[o['action']],31,GOLD,True)
    text(d,(85,890),'出力スコア 0.95以上 + コントラスト25以上 + 連続2観測の一致。観測は最大3回。',30,WHITE,True)
    text(d,(85,982),'スコアはモデルの出力であり、正解確率として校正された値ではありません。',25,MUTED)
    return im

def summary_screen():
    data=json.loads((ROOT/'results/benchmark/summary.json').read_text())
    g={(r['condition'],r['mode']):r for r in data['groups']}
    im=header('実測結果 / 効果と限界を分けて報告');d=ImageDraw.Draw(im)
    text(d,(80,210),'再読取りで見え方が改善する条件：標識に従った完走',37,WHITE,True)
    names=[('baseline_once','旧モデル・一回'),('robust_once','改良モデル・一回'),('robust_gate','改良モデル・閾値停止'),('active_recheck','改良モデル・再読取り')]
    for i,(mode,title) in enumerate(names):
        x=80+i*460;card(d,(x,310,x+425,590));text(d,(x+22,338),title,26,MUTED,True)
        text(d,(x+212,422),f"{g['variable',mode]['route_successes']} / 30",65,AQUA if i==3 else WHITE,True,anchor='ma')
    card(d,(80,635,940,895));card(d,(975,635,1845,895))
    text(d,(110,663),'全面被覆',29,GOLD,True);text(d,(110,715),'30 / 30 適切な位置で停止',39,WHITE,True)
    text(d,(110,789),'完走0件。見えないままの分岐進入0件。',25,MUTED)
    text(d,(1005,663),'見えにくさが最後まで残る条件',29,GOLD,True)
    text(d,(1005,715),'6 / 30 完走、24 / 30 停止',39,WHITE,True)
    text(d,(1005,789),'再読取りで情報が増えなければ進めない。',25,MUTED)
    text(d,(85,935),'代償：通常条件の平均走行時間 66.9秒 → 79.2秒。安全性の保証ではなく、設定内の比較結果。',25,MUTED)
    text(d,(85,994),'学習済みモデル・コード・全480試行・判定ログ・再現手順を同梱。',24,MUTED)
    return im
