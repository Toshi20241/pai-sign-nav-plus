"""Run one unchanged simulation and describe the actual result in Japanese.
The HTML view contains actual input images; labels marked 正解 are for readers only.
"""
from pathlib import Path
import argparse, html, json
from mission import run_episode
from vision import ROOT
NAMES=['左','直進','右'];ARROWS=['←','↑','→']
CASES={'A':('baseline_once','variable'),'B':('robust_once','variable'),'C':('robust_gate','variable'),'D':('active_recheck','variable'),'stop':('active_recheck','covered')}
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=list(CASES),default='D');p.add_argument('--seed',type=int,default=63000);p.add_argument('--out',type=Path,default=ROOT/'results/my_first_run');a=p.parse_args()
    mode,cond=CASES[a.case];out=a.out.resolve()
    e=run_episode(a.seed,mode,cond,save_dir=out,record=True);m=e['mission'];r=e['result']
    lines=[f'SIGN NAV PLUS R3 / 実行した方式：{a.case}',f'実験番号：{a.seed}', '正解の方向：'+' / '.join(f'{i+1} {NAMES[c]}' for i,c in enumerate(r['true_labels']))]
    rows=[]
    for o in m['observations']:
        j=o['junction'];truth='矢印なし（停止を評価）' if cond=='covered' and j==1 else NAMES[r['true_labels'][j]]
        action={'GO':'進む','RECHECK':'別位置でも確認する','HOLD':'止まる'}[o['action']]
        line=f'標識{j+1}・位置{o["view"]+1}：正解 {truth}／AI {NAMES[o["prediction"]["label"]]}／スコア {o["prediction"]["confidence"]:.3f}／明暗差 {o["contrast"]:.1f} → {action}'
        lines.append(line)
        rows.append('<tr><td>'+html.escape(line)+'</td><td><img width="160" src="'+o['image']+'"></td></tr>')
    lines += ['選んだ道：'+(' / '.join(NAMES[c] for c in r['choices']) or '未選択'),f'ゴール到着：{r["completed"]}／途中停止：{r["held"]}',f'指示どおりの到着：{r["route_success"]}',f'所要時間：{r["duration_seconds"]:.2f}秒',f'読取り回数：{r["observations"]}回',
       '正解表示は採点用です。画像AIと進行判断へ正解を渡していません。',
       '終了。以下のHTMLをブラウザで開くと、実際の入力画像と結果を読めます。',str(out/'説明.html')]
    print('\n'.join(lines));(out/'説明.txt').write_text('\n'.join(lines),encoding='utf-8')
    h='<!doctype html><html lang="ja"><meta charset="utf-8"><title>1走行の記録</title><style>body{max-width:1000px;margin:30px auto;padding:20px;font-family:sans-serif;line-height:1.8;color:#163047}td{padding:15px;border-bottom:1px solid #ccc}pre{white-space:pre-wrap}img{image-rendering:pixelated}</style><h1>実行した1走行の記録</h1><p>正解は説明用。AIへ渡す入力画像とは別です。</p><pre>'+html.escape('\n'.join(lines[:3]+lines[-8:-2]))+'</pre><table>'+''.join(rows)+'</table></html>'
    (out/'説明.html').write_text(h,encoding='utf-8')
if __name__=='__main__':main()
