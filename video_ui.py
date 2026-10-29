"""Clear Japanese explanation graphics, rendered ONLY from saved run logs.
Reference arrows and the dashed answer route are explanation overlays. They are
never passed to the classifier or controller. Robot motion comes from telemetry.
"""
from __future__ import annotations
import csv,json,math,os
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from vision import ROOT

W,H=1920,1080
BG='#F7F9FC'; WHITE='#FFFFFF'; INK='#163047'; MUTED='#506274'; LINE='#CFD8E3'
TEAL='#007D80'; TEAL_LIGHT='#E8F4F4'; ORANGE='#B94F17'; ORANGE_LIGHT='#FFF0E8'; BLUE='#235CB8'; PALE='#EDF2F8'
FONT_REG=os.environ.get('PAI_FONT_REGULAR','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
FONT_BOLD=os.environ.get('PAI_FONT_BOLD','/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
ASSETS=ROOT/'presentation/assets'
JP=['左','直進','右']; ARR=['←','↑','→']

@lru_cache(None)
def font(size,bold=False):
    path=FONT_BOLD if bold else FONT_REG
    if not Path(path).exists():
        for p in ('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc','/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc'):
            if Path(p).exists():path=p;break
    return ImageFont.truetype(path,int(size))

def text(im,xy,s,size=32,fill=INK,bold=False,anchor=None):
    ImageDraw.Draw(im).text(xy,str(s),font=font(size,bold),fill=fill,anchor=anchor)

def wrap(im,xy,s,width,size=32,fill=INK,bold=False,spacing=1.45):
    d=ImageDraw.Draw(im);f=font(size,bold);x,y=xy
    for block in str(s).split('\n'):
        line=''
        for ch in block:
            if line and d.textlength(line+ch,font=f)>width:
                text(im,(x,y),line,size,fill,bold);y+=int(size*spacing);line=ch
            else:line+=ch
        if line:text(im,(x,y),line,size,fill,bold)
        y+=int(size*spacing)
    return y

def box(im,rect,fill=WHITE,outline=LINE,radius=20,width=2):
    ImageDraw.Draw(im).rounded_rectangle(rect,radius,fill=fill,outline=outline,width=width)

def base(title,tag='SIGN NAV PLUS  /  共通提出版 R3'):
    im=Image.new('RGB',(W,H),BG)
    text(im,(55,24),tag,25,TEAL,True)
    text(im,(1865,24),'Toshigakusei',25,MUTED,anchor='ra')
    text(im,(55,74),title,49,INK,True)
    ImageDraw.Draw(im).line((55,148,1865,148),fill=LINE,width=2)
    return im

def caption(im,line1,line2=''):
    box(im,(38,920,1882,1052),INK,INK,18)
    text(im,(68,939),line1,34,WHITE,True)
    if line2:text(im,(68,990),line2,29,'#DBE8F2')

def raw(im,path,rect):
    tile=Image.open(path).convert('RGB');x,y,w,h=map(int,rect)
    tile=tile.resize((w,h),Image.Resampling.NEAREST)
    im.paste(tile,(x,y));ImageDraw.Draw(im).rectangle((x,y,x+w,y+h),outline=LINE,width=2)

class Scene:
    def __init__(self,name):
        self.name=name;self.path=ROOT/'results'/name
        self.m=json.loads((self.path/'mission.json').read_text());self.world=json.loads((self.path/'world.json').read_text())
        self.result=self.m['result'];self.duration=self.result['duration_seconds'];self.obs=self.m['observations']
        self.rows=list(csv.DictReader((self.path/'telemetry.csv').open()))
        for r in self.rows:
            for k in ('time','x','y','theta'):r[k]=float(r[k])
            for k in ('observation_index','junction','view','choice'):r[k]=int(r[k])
        self.times=np.array([r['time'] for r in self.rows]);self.xy=np.array([[r['x'],r['y']] for r in self.rows])
    def state(self,t):
        i=max(0,min(len(self.rows)-1,int(np.searchsorted(self.times,t,side='right')-1)))
        r=self.rows[i];oi=r['observation_index'];o=self.obs[oi] if oi>=0 else None
        choices=[e['choice'] for e in self.m['events'] if e['event']=='COMMIT_BRANCH' and e['time']<=t]
        return i,r,o,choices

@lru_cache(None)
def scene(name):return Scene(name)

def map_draw(im,rect,sc=None,t=None,show_answer=True,actual=True,labels=True,covered=False):
    """Uniform-scale top-down map: north is up; robot starts at the bottom."""
    x,y,w,h=rect;d=ImageDraw.Draw(im)
    sc=sc or scene('demo_active');world=sc.world
    scale=min((w-135)/4.5,(h-80)/12.4)
    cx=x+w/2+30;by=y+h-45
    def pt(px,py):return (cx+px*scale,by-(py+.45)*scale)
    d.rounded_rectangle((x,y,x+w,y+h),15,fill=WHITE,outline=LINE,width=2)
    # Draw the three available branch lanes, which merge after each junction.
    for j,yy in enumerate(world['junction_y']):
        for direction in ('left','straight','right'):
            points=[(0,yy-.35)]+world['branches'][j][direction]
            d.line([pt(*v) for v in points],fill='#C8D0DA',width=3)
        end=world['branches'][j]['straight'][-1]
        nxt=(0,world['junction_y'][j+1]-.35) if j<2 else (0,10.85)
        d.line([pt(*end),pt(*nxt)],fill='#C8D0DA',width=3)
        for station in world['stations'][j]:
            px,py=pt(*station);d.ellipse((px-3,py-3,px+3,py+3),fill=BLUE)
        px,py=pt(0,yy-.35)
        label=f'{j+1}  {ARR[sc.result["true_labels"][j]]}'
        if covered and j==1:label='2  不明'
        text(im,(x+13,py-16),label,27,INK,True)
    d.line([pt(0,-.35),pt(0,.85)],fill='#C8D0DA',width=3)
    # Pillars are fixed obstacles; no unknown obstacle avoidance is claimed.
    for o in world['obstacles']:
        if o['kind']!='pillar':continue
        a=pt(o['x']-o['width']/2,o['y']+o['height']/2);b=pt(o['x']+o['width']/2,o['y']-o['height']/2)
        d.rectangle((*a,*b),fill='#DCE1E8',outline='#B9C3CE')
    if show_answer and not covered:
        pts=[(0,-.35)]
        for j,lab in enumerate(sc.result['true_labels']):
            pts.append((0,world['junction_y'][j]-.35));pts+=world['branches'][j][('left','straight','right')[lab]]
        pts.append((0,10.85))
        for a,b in zip(pts[:-1],pts[1:]):
            pa=np.array(pt(*a));pb=np.array(pt(*b));dist=float(np.linalg.norm(pb-pa))
            if dist<1:continue
            for s in np.arange(0,dist,11):
                e=min(s+5,dist);d.line([tuple(pa+(pb-pa)*s/dist),tuple(pa+(pb-pa)*e/dist)],fill='#667584',width=3)
    if actual:
        t=sc.duration if t is None else t
        i,r,o,choices=sc.state(t)
        color=ORANGE if sc.name=='demo_baseline' else TEAL
        pts=[pt(*v) for v in sc.xy[:i+1:2]]
        if len(pts)>1:d.line(pts,fill=color,width=5)
        px,py=pt(r['x'],r['y']);th=r['theta']
        forward=np.array([math.cos(th),-math.sin(th)]);side=np.array([math.sin(th),math.cos(th)])
        center=np.array([px,py]);poly=[center+forward*14,center-forward*10+side*9,center-forward*10-side*9]
        d.polygon([tuple(v) for v in poly],fill=color,outline=INK)
    gx,gy=pt(0,10.85);d.ellipse((gx-7,gy-7,gx+7,gy+7),outline=INK,width=2)
    text(im,(gx,y+8),'ゴール',25,INK,True,anchor='ma')
    text(im,(cx,by+13),'出発',25,INK,True,anchor='ma')
    return pt

def truth_band(im,covered=False):
    box(im,(55,170,1865,247),PALE,PALE)
    s='採点用の正解：  ① 左 ←     ② 右 →     ③ 左 ←'
    if covered:s='停止試験：  ① 左 ←     ② 矢印なし（停止）     ③ 未走行'
    text(im,(79,184),s,34,INK,True)
    text(im,(1838,193),'この正解表示はAIに渡さない',25,MUTED,anchor='ra')

def current_panel(im,rect,sc,t,title):
    x,y,w,h=rect;box(im,(x,y,x+w,y+h))
    color=ORANGE if sc.name=='demo_baseline' else TEAL
    text(im,(x+20,y+15),title,31,color,True)
    text(im,(x+w-20,y+21),f'時刻 {min(t,sc.duration):05.1f} 秒',24,MUTED,anchor='ra')
    map_draw(im,(x+15,y+70,360,h-110),sc,t,covered=sc.result['condition']=='covered')
    i,r,o,choices=sc.state(t);sx=x+398;sw=w-422
    if o:
        text(im,(sx,y+80),f'直近の入力：標識{o["junction"]+1}・位置{o["view"]+1}',27,INK,True)
        raw(im,sc.path/o['image'],(sx+58,y+132,208,208))
        text(im,(sx,y+355),f'AI判定：{JP[o["prediction"]["label"]]} {ARR[o["prediction"]["label"]]}',38,color,True)
        text(im,(sx,y+410),f'出力スコア：{o["prediction"]["confidence"]:.3f}',26,MUTED)
        action={'GO':'進路を確定','RECHECK':'まだ進路を確定しない','HOLD':'この先へ進まず停止'}[o['action']]
        wrap(im,(sx,y+455),action,sw,28,INK,True)
    else:
        wrap(im,(sx,y+125),'最初の観測位置へ移動中',sw,34,INK,True)
        wrap(im,(sx,y+225),'停止し、向きを整えてから画像を受け取る。',sw,28,MUTED)
    chosen=' / '.join(f'{j+1}:{JP[c]}' for j,c in enumerate(choices)) or 'まだ選んでいません'
    text(im,(sx,y+h-134),'確定した道',25,MUTED)
    wrap(im,(sx,y+h-94),chosen,sw,29,INK,True)
    if t>=sc.duration:
        if sc.result['held']:end='停止完了'
        elif sc.result['route_success']:end='指示どおりに到着'
        else:end='到着したが、③の指示違反'
        box(im,(x+16,y+h-44,x+w-16,y+h-4),TEAL_LIGHT if sc.result['scenario_success'] else ORANGE_LIGHT,None,8)
        text(im,(x+w/2,y+h-44),end,25,color,True,anchor='ma')

def paired_frame(t):
    im=base('走行を見る：正解の道と、実際に選んだ道を比べる')
    truth_band(im)
    current_panel(im,(55,266,891,637),scene('demo_baseline'),t,'A 基本AI ＋ 1回で決定')
    current_panel(im,(974,266,891,637),scene('demo_active'),t,'D 強化AI ＋ 再確認')
    caption(im,'点線＝指示どおりの道　色線＝記録された走行　／　走行は2倍速',
            '2つの走行を同時に開始。読取り中の「標識番号・位置番号」が異なる場合があります。')
    return im

def hold_frame(t):
    sc=scene('demo_hold');im=base('読めない場合：②の分岐に入る前で止まる')
    truth_band(im,True)
    current_panel(im,(55,266,891,637),sc,t,'D 強化AI ＋ 再確認')
    box(im,(974,266,1865,903))
    text(im,(1000,287),'②の各位置で実際に渡した画像',33,INK,True)
    for v in range(3):
        xx=1001+v*279;raw(im,sc.path/f'camera/j2_v{v+1}.png',(xx+33,364,200,200))
        text(im,(xx+133,582),f'位置{v+1}',30,INK,True,anchor='ma')
        text(im,(xx+133,636),'矢印は描かない',25,MUTED,anchor='ma')
    wrap(im,(1003,706),'絵の明暗差は7。必要な25に届かず、\n3回目でも確認できないため停止。',807,33,INK,True)
    caption(im,'これは「矢印なし」の設定。見えない方向を当てる試験ではありません。',
            '走行記録の後半を1倍速で再生。停止は「指示を守った到着」と別に数えます。')
    return im

def title_card():
    im=base('SIGN NAV PLUS：矢印を読み、確認してから進む')
    text(im,(90,213),'何を試した作品か',37,TEAL,True)
    wrap(im,(90,277),'画像AIの答えを、そのまま行動に使うか。\nもう一度確かめてから、動くか。',1730,59,INK,True,1.48)
    for i,(head,desc) in enumerate([('1 見る','合成した矢印をAIで分類'),('2 確かめる','別の位置の画像でも確認'),('3 動く／止まる','条件がそろわなければ停止')]):
        x=70+i*604;box(im,(x,535,x+572,820))
        text(im,(x+28,565),head,41,TEAL,True)
        wrap(im,(x+28,652),desc,516,36,INK)
    caption(im,'実カメラ・実機ではなく、合成画像と平面移動のシミュレーションです。',
            '正解・AIに見せた画像・選んだ道を区別して説明します。')
    return im

def rule_card():
    im=base('最初にルール：ゴールに着くだけでは成功にしない')
    truth_band(im);map_draw(im,(90,270,620,627),scene('demo_active'),actual=False)
    text(im,(780,282),'この例の正解は「左 → 右 → 左」',40,TEAL,True)
    wrap(im,(780,362),'画面の下から上へ進みます。\n← は左側の枝、↑ は中央、→ は右側です。',1060,35)
    box(im,(765,515,1850,700),ORANGE_LIGHT,None)
    wrap(im,(795,539),'どの枝も、その先で合流します。\n間違った道でもゴールには着けます。',1000,39,INK,True)
    wrap(im,(780,746),'成功＝3つの指示を全部守り、\n接触せずにゴールへ到着すること。',1050,37,INK,True)
    caption(im,'正解は、画像を作る前に環境側で決めます。AIの回答を正解にはしません。',
            '点線は説明用の正解経路です。AIや移動制御に正解ラベルは渡しません。')
    return im

def data_card():
    im=base('900枚はどこから？　写真ではなく、プログラムで描く')
    for lab in range(3):
        x=96+lab*270;raw(im,ASSETS/f'baseline_train_{lab}.png',(x,223,190,190))
        text(im,(x+95,427),f'{JP[lab]} 300枚',30,INK,True,anchor='ma')
    box(im,(920,202,1855,496),TEAL_LIGHT,None)
    wrap(im,(954,228),'基本AIの学習：900枚\n左・直進・右を各300枚',861,42,INK,True)
    text(im,(952,393),'強化AIの学習：4,500枚',38,TEAL,True)
    box(im,(75,543,1855,875))
    text(im,(108,568),'「見えにくい画像900枚」は、別に作ったテスト用',39,INK,True)
    wrap(im,(109,644),'回転・ぼかし・ノイズ・一部を隠す加工を加えます。\n同じ900枚を両方のAIに見せ、方向が合うかを採点します。',1660,35)
    text(im,(109,773),'基本AI：421 / 900枚　　強化AI：798 / 900枚',40,TEAL,True)
    caption(im,'学習用900枚と、テスト用900枚は別の画像です。900回の走行ではありません。',
            '表示例は実際の学習データを同じ生成手順で再現したものです。')
    return im

def environment_card():
    im=base('読み直しの設定：場所ごとに、受け取る合成画像を決める')
    for v in range(3):
        x=75+v*610;box(im,(x,201,x+570,775))
        text(im,(x+285,225),f'標識③・位置{v+1}',36,INK,True,anchor='ma')
        raw(im,ASSETS/f'demo_active_j3_v{v+1}.png',(x+160,307,250,250))
        wrap(im,(x+28,594),'見えにくい画像' if v==0 else '比較的見やすい画像',514,36,TEAL,True)
        wrap(im,(x+28,657),'強い加工の設定' if v==0 else '弱い加工の設定',514,31,MUTED)
    text(im,(84,821),'同じ実験番号・標識・位置なら、方式を変えても同じ画像です。',34,INK,True)
    caption(im,'位置を変えると、あらかじめ決めた画像条件に切り替わる実験です。',
            'カメラの移動によって自然に鮮明になる様子を、光学的に計算したものではありません。')
    return im

def third_card():
    im=base('③を拡大：同じ画像でも、AIの答えは違った')
    box(im,(55,177,1865,257),PALE,PALE)
    text(im,(84,191),'③の正解：左 ←　（環境側が決めた採点用の答え）',36,INK,True)
    raw(im,ASSETS/'demo_active_j3_v1.png',(105,319,280,280))
    text(im,(245,613),'最初の入力画像',29,INK,True,anchor='ma')
    box(im,(449,295,1098,689))
    text(im,(474,315),'A 基本AI',34,ORANGE,True)
    text(im,(474,370),'「直進」→ そのまま進む（誤り）',31,INK,True)
    text(im,(474,446),'B 強化AI・1回判定',32,TEAL,True)
    text(im,(474,497),'「左」→ この例では正解',31,INK,True)
    text(im,(474,572),'D 強化AI・再確認',32,TEAL,True)
    text(im,(474,623),'「左」→ まだ確定しない',31,INK,True)
    for v in (2,3):
        xx=1152+(v-2)*352
        raw(im,ASSETS/f'demo_active_j3_v{v}.png',(xx+27,337,235,235))
        text(im,(xx+146,589),f'D 位置{v}：左',29,TEAL,True,anchor='ma')
    wrap(im,(1148,667),'2回とも条件を満たす「左」。\nここで左の道を確定。',671,34,INK,True)
    text(im,(78,824),'Dの確定条件：スコア0.95以上 ＋ 明暗差25以上 ＋ 直近2回の一致',33,INK,True)
    caption(im,'この1例では、再確認をしないBも成功しています。',
            '再確認の効果は、同じ強化AIを使うBとDを、30走行で比較して判断します。')
    return im

def results_card():
    im=base('効果の比較：同じ強化AIを使い、決め方を変える')
    text(im,(70,184),'位置を変えると見やすくなる条件　／　各方式30走行',36,INK,True)
    rows=[('A 基本AI ＋ 1回','2 / 30'),('B 強化AI ＋ 1回','18 / 30'),('C 強化AI ＋ 条件未達で停止','6 / 30'),('D 強化AI ＋ 再確認','30 / 30')]
    box(im,(65,253,1848,740))
    text(im,(98,272),'方式',33,MUTED,True);text(im,(1799,272),'3つの指示を守って到着',33,MUTED,True,anchor='ra')
    for i,(a,b) in enumerate(rows):
        y=343+i*93
        if i==3:box(im,(83,y-8,1831,y+75),TEAL_LIGHT,None,8)
        text(im,(104,y),a,37,INK,True);text(im,(1782,y),b,46,TEAL if i==3 else INK,True,anchor='ra')
    wrap(im,(80,777),'見えにくさが続く条件では、Dは6 / 30到着、24 / 30停止。\n読めないまま繰り返すだけでは、到着は増えません。',1750,34,INK,True)
    caption(im,'全体は4条件 × 30実験番号 × 4方式＝480走行。停止を到着に足していません。',
            '動画の1例だけでなく、失敗した走行も含めて集計しています。')
    return im

def limits_card():
    im=base('分かったことと、この実験で確かめていないこと')
    box(im,(65,191,1855,430),TEAL_LIGHT,None)
    text(im,(98,212),'分かったこと',36,TEAL,True)
    wrap(im,(98,279),'見やすい別画像を得られる設定では、再確認が役立った。\n確認できないときは、分岐へ入る前に止まる動作を実装した。',1715,38,INK,True)
    box(im,(65,458,1855,691))
    text(im,(98,480),'追加でかかる時間',36,INK,True)
    wrap(im,(98,543),'通常条件の平均：1回判定66.9秒 → 再確認79.2秒。\n確認のため、約12.3秒余分にかかった。',1700,37)
    wrap(im,(83,735),'実カメラ・実機・未知の道は未検証。\nこの成績を、実世界の安全性や認識性能の保証とはしない。',1740,35,INK,True)
    caption(im,'成果は「認識 → 再確認 → 進行／停止」を、条件を明示して比較したことです。',
            '学習・入力画像・走行記録・実装をPDFとコードに対応付けています。')
    return im

CARDS={'intro':title_card,'rule':rule_card,'data':data_card,'environment':environment_card,
       'third':third_card,'results':results_card,'limits':limits_card}

def export_figures():
    ASSETS.mkdir(parents=True,exist_ok=True)
    for name,fn in CARDS.items():fn().save(ASSETS/f'video_{name}.png')
    paired_frame(101.05).save(ASSETS/'video_pair_final.png')
    paired_frame(53.25).save(ASSETS/'video_pair_mid.png')
    hold_frame(49.6).save(ASSETS/'video_hold_final.png')
    # Replace the previous public figure paths as well, so GitHub does not retain
    # old explanatory screenshots under the same filenames after an update.
    import shutil
    old_paths=ROOT/'results/figures';old_paths.mkdir(parents=True,exist_ok=True)
    for public_name,asset_name in {'intro':'video_intro','comparison':'video_pair_final',
                                   'hold':'video_hold_final','observations':'video_third',
                                   'summary':'video_results'}.items():
        shutil.copyfile(ASSETS/(asset_name+'.png'),old_paths/(public_name+'.png'))
    # Large portrait diagrams usable in the PDF without tiny video labels.
    for name,sc,actual in [('course',scene('demo_active'),False),('route_A',scene('demo_baseline'),True),('route_B',scene('demo_once'),True),('route_D',scene('demo_active'),True),('route_hold',scene('demo_hold'),True)]:
        im=Image.new('RGB',(610,880),WHITE)
        map_draw(im,(4,4,602,872),sc,actual=actual,covered=sc.result['condition']=='covered')
        im.save(ASSETS/f'{name}.png')
    # Shared ground-truth arrow references are explicitly different from sensor inputs.
    return ASSETS

if __name__=='__main__':export_figures()
