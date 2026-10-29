"""R3 readable submission video (178 s) and a separate slower learning video.
Motion is read from saved telemetry; answer overlays are for viewers only.
"""
from pathlib import Path
import argparse, json, subprocess, shutil
from PIL import Image, ImageDraw
import video_ui as u
from vision import ROOT
ASSETS=u.ASSETS

def base(title):return u.base(title,tag='SIGN NAV PLUS  /  共通提出版 R3')

def simple_card(title,blocks,foot1,foot2=''):
    im=base(title)
    height=min(220, int(710/len(blocks))-18)
    for i,(head,body) in enumerate(blocks):
        y=185+i*(height+20)
        u.box(im,(65,y,1855,y+height),u.TEAL_LIGHT if i==0 else u.WHITE,None)
        u.text(im,(96,y+17),head,36,u.TEAL,True)
        u.wrap(im,(96,y+78),body,1718,38,u.INK,False,1.45)
    u.caption(im,foot1,foot2)
    return im

def intro():
    return simple_card('矢印を読み、確認してから進む',[('作ったもの','合成した矢印を読むAIと、移動・停止のプログラム。'),('調べたこと','一度の答えで進むか。別位置でも確認して進むか。')],
       '実カメラ・実機ではなく、合成画像と平面移動のシミュレーション。','以降の「正解」は説明用の表示です。走行判断へ正解は渡しません。')

def ways():
    im=base('画像AIは2種類。走り方は4方式。')
    rows=[('A','基本AI','1回読み、その答えで進む。'),('B','強化AI','1回読み、その答えで進む。'),('C','強化AI','1回読み、確認条件を満たさなければ止まる。'),('D','強化AI','別位置でも読み、条件を満たす2回の一致で進む。')]
    for i,(a,b,c) in enumerate(rows):
        y=202+i*142
        u.box(im,(65,y,1855,y+122),u.TEAL_LIGHT if a=='D' else u.WHITE)
        u.text(im,(100,y+28),a,46,u.TEAL,True);u.text(im,(195,y+35),b,36,u.INK,True)
        u.text(im,(450,y+35),c,35,u.INK)
    u.caption(im,'A対B：画像AIを強化した比較。B対D：同じ強化AIで、判断方法を比較。',
              '最大3回で確認できないDは停止。走行制御をAIが学習した作品ではありません。')
    return im

def decision():
    im=base('同じ「左」でも、条件を満たすまで進まない')
    u.text(im,(70,183),'動画例の③：正解は左 ←　／　Dの実際の入力と判定',36,u.INK,True)
    rows=[('位置1','0.743','再確認'),('位置2','約1.000','再確認'),('位置3','約1.000','左へ進む')]
    for i,(a,b,c) in enumerate(rows):
        x=65+i*605;u.box(im,(x,250,x+580,740))
        u.text(im,(x+290,270),a,34,u.INK,True,anchor='ma')
        u.raw(im,ASSETS/f'demo_active_j3_v{i+1}.png',(x+180,330,220,220))
        u.text(im,(x+28,580),'判定：左／スコア '+b,31,u.INK,True)
        u.text(im,(x+28,647),c,39,u.TEAL,True)
    u.text(im,(78,779),'各回：スコア0.95以上 ＋ 明暗差25以上。直近2回が同じ方向なら進む。',31,u.INK,True)
    u.caption(im,'強化AIのBは、最初の画像だけで左を選び、この例では成功しています。',
        'Dは最初から左と答えています。読み直しで初めて左と分かった例ではありません。')
    return im

def compact_panel(im,x,sc,t,title):
    y=259;w=589;h=640;cl=u.ORANGE if sc.name=='demo_baseline' else (u.BLUE if sc.name=='demo_once' else u.TEAL)
    u.box(im,(x,y,x+w,y+h));u.text(im,(x+18,y+15),title,31,cl,True)
    u.map_draw(im,(x+10,y+75,280,490),sc,t)
    i,r,o,choices=sc.state(t);sx=x+303
    if o:
        u.text(im,(sx,y+81),f'直近：標識{o["junction"]+1}',26,u.INK,True)
        u.text(im,(sx,y+122),f'観測位置{o["view"]+1}',26,u.MUTED)
        u.raw(im,sc.path/o['image'],(sx+18,y+173,220,220))
        truth=sc.result['true_labels'][o['junction']]
        u.text(im,(sx,y+410),'正解：'+u.JP[truth]+' '+u.ARR[truth],31,u.INK,True)
        u.text(im,(sx,y+457),'AI：'+u.JP[o['prediction']['label']]+' '+u.ARR[o['prediction']['label']],34,cl,True)
        u.text(im,(sx,y+509),f'スコア {o["prediction"]["confidence"]:.3f}',25,u.MUTED)
    else:u.wrap(im,(sx,y+175),'画像を受け取る\n地点へ移動中',260,30,u.INK,True)
    selected=' / '.join(f'{j+1}{u.ARR[c]}' for j,c in enumerate(choices)) or '未確定'
    u.text(im,(x+18,y+565),'選んだ道：'+selected,27,u.INK,True)
    if t>=sc.duration:
        message='到着／③は指示違反' if not sc.result['route_success'] else '3つの指示を守って到着'
    else:message={'READ':'画像を読み、判断中','REPOSITION':'別の観測位置へ移動中','ALIGN':'停止して向きを調整中','FOLLOW_BRANCH':'確定した枝を走行中','GOAL':'ゴールへ移動中'}.get(r['state'],'観測位置へ移動中')
    u.box(im,(x+10,y+h-40,x+w-10,y+h-4),u.ORANGE_LIGHT if sc.name=='demo_baseline' and t>=sc.duration else u.PALE,None,7)
    u.text(im,(x+w/2,y+h-40),message,25,cl,True,anchor='ma')

def run_frame(t,speed=2):
    im=base('同じ標識で比較：①左、②右、③左が正解')
    u.box(im,(55,170,1865,238),u.PALE,None)
    u.text(im,(78,180),f'下から上へ走行　点線＝指示どおりの道　色線＝実際の走行　時刻 {t:5.1f}秒',32,u.INK,True)
    for x,n,title in zip((55,665,1275),('demo_baseline','demo_once','demo_active'),('A 基本AI・1回判定','B 強化AI・1回判定','D 強化AI・再確認')):
        compact_panel(im,x,u.scene(n),t,title)
    u.caption(im,f'記録された走行を{speed}倍速で再生。Bも成功する例なので、Dだけの効果とはしません。',
         '同時に開始しています。再確認するDは時間を使うため、読取り中の標識が左右で違います。')
    return im

def stop_frame(t,speed=2):
    im=u.hold_frame(t)
    # Reuse the saved-log display; only relabel its replay speed.
    u.caption(im,'②では、3か所とも矢印を描かない。3回確認して、分岐の手前で停止。',
         (f'停止した位置と判断を静止表示（直前の走行は{speed}倍速）。' if t>=49.6 else f'記録の後半を{speed}倍速で再生。停止30回は、完走30回ではありません。'))
    return im

def counts():
    return simple_card('480走行と1,673回は、単位が違う',[
        ('480走行','4つの走り方 × 4つの環境 × 30通り。\n1走行＝出発から到着、または途中停止まで。'),
        ('1,673回の読取り','その480走行の途中で、画像を読んだ延べ回数。\nDの動画例は、①3回＋②2回＋③3回＝8回。'),
        ('正解数とは別','回数を多く試したことと、何回正しくできたかは別です。')],
        'A 360回 ＋ B 360回 ＋ C 258回 ＋ D 695回 ＝ 1,673回。','同じ画像を別方式で読む場合も数えます。異なる写真1,673枚という意味ではありません。')

def audits():
    return simple_card('保存記録の点検と、再走行は別',[
        ('全480走行の記録を点検','画像を再生成 → 同じAIで判定 → 記録の答え・判断と比較。\n選んだ道と正解を比べ直し、表の合計も足し直す。'),
        ('代表の4走行だけ、最初から再実行','A、B、Dの到着例と、Dの停止例。\n選んだ道・所要時間・最後の位置などを比較。'),
        ('同じ誤答が出ても「記録は一致」','正解「左」に今回も「直進」なら、AIの答えは誤りのまま。')],
        '480走行すべてを最初から再走行した、という説明ではありません。','点検は制作環境で実施。利用者のMacやColabでの実行確認とは区別します。')

def weights():
    return simple_card('同じ問題を作り直す／同じAIを作り直す',[
        ('テスト画像の作り直し','元と同じ設定で900枚ずつを再生成し、両AIに読ませる。\n報告した正解数421枚・798枚などを再確認。'),
        ('基本AIの再学習','学習用900枚と正解を使い、元と同じ60周を学習。\n保存済みのAIと、内部の調整値「重み」が一致。'),
        ('点検の意味','同梱手順で結果や基本AIを作り直せるかの確認。\n再学習して、さらに賢くしたという意味ではありません。')],
        '学習用900枚とテスト用900枚は別。強化AIの再学習点検までは行っていません。','乱数の開始設定seedを固定し、同じ生成条件を再現しています。')

def unit_tests():
    return simple_card('27件のチェックは、ルールの動作確認',[
        ('例1：答えを決めて返す部品を使う','「左、左」と高スコアを順に返す。\n1回目は再確認、2回目は進む処理になるか調べる。'),
        ('例2：そろわなければ止まる','3回の答えがそろわない場合、停止するか調べる。\n同じ画像の再現や、車輪の加速度上限も点検。'),
        ('確認できる範囲','指定した処理が、用意した点検条件で動いた。\n画像認識の全問正解や、実機の安全性とは別です。')],
        '27件すべて通過。数字の多さを成果とせず、何を点検したかを示します。','全項目と日本語説明は、コードZIPの docs/test_checklist.md に対応しています。')

def end():
    return simple_card('説明するときは、この順で',[('1 何を作ったか','画像分類AIと、進む・再確認する・止まる処理。'),('2 どう比べたか','同じ条件で4方式を比較。画像の正解枚数と走行結果を分ける。'),('3 何が分かり、どこまでか','追加画像が見やすい条件では有効。読めなければ止まる。\n実カメラ・実機・未知の道での性能は未検証。')],
      '通常課題と優秀生選出用で、同じR3のPDF・動画・コードを使います。','手元の理解用動画です。3分以内の提出動画とは別ファイルです。')

CARDS={'intro':intro,'rule':u.rule_card,'data':u.data_card,'ways':ways,'environment':u.environment_card,
 'decision':decision,'results':u.results_card,'limits':u.limits_card,'counts':counts,'audits':audits,'weights':weights,'tests':unit_tests,'end':end}
SUBMIT=[('intro',9),('rule',12),('data',14),('ways',12),('environment',14),('run',51),('decision',12),('stop',15),('results',22),('limits',17)]
LEARN=[('intro',18),('rule',30),('data',32),('ways',32),('environment',30),('run',102),('decision',30),('stop',30),('results',32),('limits',30),('counts',32),('audits',32),('weights',32),('tests',32),('end',16)]

def export():
    ASSETS.mkdir(parents=True,exist_ok=True)
    if not (ASSETS/'baseline_train_0.png').exists():
        from make_explanation_assets import main as prepare_assets
        prepare_assets()
    for name,fn in CARDS.items():fn().save(ASSETS/f'r3_{name}.png')
    run_frame(101.05).save(ASSETS/'r3_run_end.png');run_frame(53.25).save(ASSETS/'r3_run_middle.png');stop_frame(49.6).save(ASSETS/'r3_stop_end.png')

def render(out,kind='submit'):
    from report_guard import assert_report_snapshot
    assert_report_snapshot();export();fps=20 if kind=='submit' else 12; speed=2 if kind=='submit' else 1
    sections=SUBMIT if kind=='submit' else LEARN;out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),'-i','-',
           '-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    timeline=[];n=0
    try:
        for name,seconds in sections:
            begin=n/fps
            if name not in ('run','stop'):
                data=Image.open(ASSETS/f'r3_{name}.png').convert('RGB').tobytes()
                for _ in range(round(seconds*fps)):proc.stdin.write(data)
            else:
                for i in range(round(seconds*fps)):
                    t=min(i/fps*speed,101.05) if name=='run' else min((29.6 if speed==2 else 24.6)+i/fps*speed,49.6)
                    im=run_frame(t,speed) if name=='run' else stop_frame(t,speed)
                    proc.stdin.write(im.tobytes())
            n+=round(seconds*fps);timeline.append({'section':name,'start':begin,'end':n/fps,'playback':speed if name in ('run','stop') else 'still'})
            print(kind,name,n/fps,flush=True)
    finally:proc.stdin.close()
    if proc.wait():raise RuntimeError('ffmpeg rendering failed')
    meta={'edition':'R3','kind':kind,'duration':n/fps,'fps':fps,'width':1920,'height':1080,'audio':'none; Japanese on-screen explanation','sections':timeline}
    (ROOT/f'results/video_r3_{kind}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    return meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'deliverables/動画_PAI最終課題_Toshigakusei.mp4');p.add_argument('--kind',choices=['submit','learn'],default='submit');p.add_argument('--preview',action='store_true');a=p.parse_args()
    if a.preview:export()
    else:render(a.out,a.kind)
if __name__=='__main__':main()
