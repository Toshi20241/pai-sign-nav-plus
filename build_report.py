"""Build a four-page Japanese PDF from the recorded experiment outputs.

Optional PAI_FONT_REGULAR/PAI_FONT_BOLD point to installed TrueType fonts.
Without those, ReportLab's standard Japanese CID font is used. No fonts bundled.
"""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from vision import ROOT

PAGE_W,PAGE_H=A4
NAVY=colors.HexColor('#142B41');TEAL=colors.HexColor('#007C78');INK=colors.HexColor('#253C4C');GRAY=colors.HexColor('#536777');PALE=colors.HexColor('#F0F5F7')
FONT='JP';BOLD='JPBold'

def register_fonts():
    global FONT,BOLD
    a=os.environ.get('PAI_FONT_REGULAR');b=os.environ.get('PAI_FONT_BOLD')
    if a and Path(a).is_file():
        pdfmetrics.registerFont(TTFont('JP',a));FONT='JP'
        pdfmetrics.registerFont(TTFont('JPBold',b or a));BOLD='JPBold'
    else:
        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'));FONT=BOLD='HeiseiKakuGo-W5'
    pdfmetrics.registerFontFamily(FONT,normal=FONT,bold=BOLD,italic=FONT,boldItalic=BOLD)

def para(c,s,x,y,w=499,size=10.2,leading=None,color=INK,bold=False):
    st=ParagraphStyle('text',fontName=BOLD if bold else FONT,fontSize=size,leading=leading or size*1.60,
                      textColor=color,wordWrap='CJK',spaceAfter=0)
    p=Paragraph(s,st);_,h=p.wrap(w,1000);p.drawOn(c,x,y-h);return y-h

def head(c,page,kicker,title,subtitle):
    c.setFillColor(TEAL);c.setFont(BOLD,8);c.drawString(48,PAGE_H-32,'SIGN NAV +  /  PHYSICAL AI')
    c.setFillColor(GRAY);c.setFont(FONT,8);c.drawRightString(PAGE_W-48,PAGE_H-32,'Toshigakusei')
    c.setStrokeColor(colors.HexColor('#D8E3E8'));c.line(48,PAGE_H-45,PAGE_W-48,PAGE_H-45)
    y=PAGE_H-69;y=para(c,kicker,48,y,size=8.5,color=TEAL,bold=True)
    y=para(c,title,48,y-9,size=24,leading=32,color=NAVY,bold=True)
    y=para(c,subtitle,48,y-10,size=10,color=GRAY)
    c.setFont(FONT,8);c.setFillColor(GRAY);c.drawString(48,27,'合成標識・既知コース・平面運動シミュレーション')
    c.drawRightString(PAGE_W-48,27,f'{page} / 4')
    return y-22

def h2(c,s,y):
    c.setFillColor(TEAL);c.rect(48,y-13,3,13,fill=1,stroke=0)
    return para(c,s,60,y+1,w=487,size=12.5,color=NAVY,bold=True)-10

def table(c,rows,y,widths,header=True,size=9.2):
    body=[]
    for i,row in enumerate(rows):
        cells=[]
        for cell in row:
            cells.append(Paragraph(str(cell),ParagraphStyle('cell',fontName=BOLD if i==0 and header else FONT,
                fontSize=size,leading=size*1.48,textColor=colors.white if i==0 and header else INK,wordWrap='CJK')))
        body.append(cells)
    t=Table(body,colWidths=widths,hAlign='LEFT')
    style=[('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
           ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,0),.5,NAVY)]
    if header:style += [('BACKGROUND',(0,0),(-1,0),NAVY)]
    for i in range(1 if header else 0,len(rows)):
        style.append(('BACKGROUND',(0,i),(-1,i),PALE if i%2 else colors.white))
    t.setStyle(TableStyle(style));_,h=t.wrap(499,1000);t.drawOn(c,48,y-h);return y-h-13

def box(c,title,body,y):
    st=ParagraphStyle('box',fontName=FONT,fontSize=10,leading=16,wordWrap='CJK',textColor=INK)
    p=Paragraph(body,st);_,h=p.wrap(467,1000);height=h+50
    c.setFillColor(PALE);c.roundRect(48,y-height,499,height,8,fill=1,stroke=0)
    para(c,title,64,y-13,w=467,size=11.2,color=TEAL,bold=True);p.drawOn(c,64,y-height+13)
    return y-height-15

def main(out):
    from report_guard import assert_report_snapshot
    assert_report_snapshot()
    register_fonts()
    data=json.loads((ROOT/'results/benchmark/summary.json').read_text());groups={(r['condition'],r['mode']):r for r in data['groups']}
    vis=json.loads((ROOT/'results/vision_metrics.json').read_text());audit=json.loads((ROOT/'results/verification.json').read_text())
    c=canvas.Canvas(str(out),pagesize=A4);c.setTitle('SIGN NAV + | 読み直して進む。読めなければ止まる。');c.setAuthor('Toshigakusei')
    y=head(c,1,'01 / TASK & IDEA','読み直して進む。\n読めなければ止まる。'.replace('\n','<br/>'),
           '矢印標識を読む移動ロボット：一回判定から、再観測と判断保留へ')
    y=para(c,'架空の三分岐コースで、ロボットが矢印画像を「左・直進・右」に分類し、指定された枝を進む。読み取りが曖昧なまま動くのではなく、別の観測位置へ移動して画像を読み直す。確認できなければ、分岐に入る前に停止する。',48,y)
    y-=17
    img=ROOT/'results/figures/comparison.png';h=499*1080/1920
    c.drawImage(str(img),48,y-h,width=499,height=h);y-=h+8
    y=para(c,'同じseed 63000の比較例。左は旧モデルの一回判定、右は改良モデルの再読取り。軌跡は車輪状態を積分した走行ログから描画。動画の走行部分は2倍速。',48,y,size=8.3,color=GRAY)
    y=h2(c,'今回加えた工夫',y-20)
    y=table(c,[['認識を強める','観測を増やす','行動を抑制する'],
        ['強い回転・ぼかし・部分被覆を含む合成画像で学習。','停止後、別位置で再読取り。連続2回の一致を確認。','最大3回で打ち切り。判定条件を満たさなければ停止。']],y,[166,166,167],size=9.5)
    y=para(c,'目的は、画像分類の正解率だけでなく「正しい枝を選べたか」「停止しすぎないか」「余計に時間がかからないか」を、同じ条件で測ること。',48,y,size=10.1)
    c.showPage()
    y=head(c,2,'02 / IMPLEMENTATION','AIと車輪制御を分ける','正解ラベルは環境と評価器のみが保持。判断部の入力は画像だけ。')
    y=h2(c,'処理の流れ',y)
    y=box(c,'画像 → 分類 → 確認 → 移動／停止','96×96画素の合成標識を24×24画素へ縮小し、MLPで3分類。<br/>出力スコアと画素コントラストを確認し、直近2観測が同じ方向なら進路を確定する。再読取りは既知の観測位置へ移動して行う。3回でも確認できなければHOLDで停止する。',y)
    y=table(c,[['項目','実装と固定条件'],
        ['画像AI','旧：576→32→3、学習900枚。改良：576→64→32→3、学習4,500枚・80エポック。ReLU／Adam。'],
        ['学習と評価','学習seed 51001、確認用600枚は52001、独立画像テスト各900枚は53001／54001。学習画像の再利用なし。'],
        ['判定条件','出力スコア0.95以上、画像コントラスト25以上、直近2回の一致。最大3観測。値はテスト前に固定。'],
        ['左右輪の制御','既知の経由点へのフィードバック。車輪の一次遅れ0.12秒、角加速度上限32 rad/s²を含む。'],
        ['時刻と停止','積分0.01秒・制御0.05秒。読み取りは停止・整列後。停止指令後も車輪が減速する過程を計算。'],
        ['センサーの前提','位置ごとの合成標識パッチ。遠近投影や実カメラの標識検出は未実装。自己位置・コース形状は既知。']],y,[96,403],size=9.2)
    y=h2(c,'位置を変えることで、何が変わるか',y-7)
    y=para(c,'再読取りで改善する条件では、最初の位置を強い画像変動、残り2位置を通常の変動としている。センサー出力はseed・標識番号・観測位置で固定され、方式によって都合よく変更しない。全面被覆の標識は、どの位置からも矢印が見えない。',48,y)
    y=para(c,'出力スコアは校正済みの正解確率ではない。コントラスト検査も、全面被覆という今回の条件を扱う簡単な画像品質チェックであり、一般的な未知画像の検出器ではない。',48,y-13,size=9.4,color=GRAY)
    c.showPage()
    y=head(c,3,'03 / PAIRED EVALUATION','効果を、4方式で切り分ける','4条件 × 各30seed × 4方式 = 480走行。成功したseedだけに絞らず全件集計。')
    y=h2(c,'画像の独立テスト',y)
    y=table(c,[['画像条件','旧モデル','改良モデル'],['通常画像：900枚','900 / 900（100.0%）','900 / 900（100.0%）'],
        ['強い画像変動：900枚','421 / 900（46.8%）','798 / 900（88.7%）']],y,[215,142,142],size=9.5)
    y=h2(c,'全3標識を守って完走した件数（各30走行）',y-4)
    y=table(c,[['方式','通常','再読取りで改善','変動が持続'],
        ['旧モデル・一回判定','30','2','2'],['改良モデル・一回判定','30','18','18'],
        ['改良モデル・閾値停止','30','6','6'],['改良モデル・再読取り','30','30','6']],y,[205,78,116,100],size=9.5)
    y=para(c,'持続変動条件は、全観測位置で強い変動が残る追加の限界確認。モデル・閾値は変更していない。再読取り方式は6件完走、24件判断保留。単に回数を増やしても、観測が十分に改善しなければ完走は増えなかった。',48,y,size=9.5)
    y=h2(c,'2番目の矢印を全面被覆した場合（各30走行）',y-18)
    y=table(c,[['方式','適切な位置で停止','読めない分岐へ進入'],
        ['旧モデル・一回判定','0 / 30','30 / 30'],['改良モデル・一回判定','0 / 30','30 / 30'],
        ['改良モデル・閾値停止','18 / 30','0 / 30'],['改良モデル・再読取り','30 / 30','0 / 30']],y,[205,147,147],size=9.5)
    y=para(c,'「適切な位置で停止」は、1番目の指示を守り、被覆された2番目の分岐前で止まること。閾値停止方式の残り12件は1番目前で止まった。再読取り方式の停止30件は完走0件であり、完走率には加算していない。',48,y,size=9.1,color=GRAY)
    y=para(c,'出典：results/benchmark/{protocol.json, trials.csv, trials.json, summary.json}。同一seedでは標識・初期位置・車輪係数・同じ位置の入力画像を共通化。',48,y-10,size=8.1,color=GRAY)
    c.showPage()
    y=head(c,4,'04 / LIMITS & REPRODUCIBILITY','成功だけでなく、代償も残す','「止まれる」ことと「目的地へ進める」ことを、別の評価として扱う。')
    y=h2(c,'結果の解釈',y)
    y=box(c,'速度との交換条件','全方式が完走した通常条件でも、再読取り方式の平均時間は79.2秒。一回判定の66.9秒より約12.3秒長い。認識の確認を増やす代わりに、移動と待機の時間を使っている。',y)
    y=para(c,'再読取りで改善する条件では、改良モデルの一回判定18/30に対し、再読取り30/30。閾値を設けるだけでは6/30に低下したため、今回は「疑わしい出力を止める」ことに加えて、見え方を改善できる別位置へ動くことが有効だった。',48,y)
    y=h2(c,'この実験で言える範囲',y-19)
    y=para(c,'固定コース、正確な自己位置、観測位置が既知という条件での結果である。接触0件はこの経路・幾何判定内の観測であり、実機の衝突回避性能を示さない。滑り・実カメラ投影・未知障害物・実画像への適用は検証していない。',48,y)
    y=para(c,'MLPは画像分類用の教師あり学習。移動制御は学習していない。モデル容量と学習画像の両方を変更したため、画像精度改善への個別の寄与は未分離。30/30の成功でも、Wilson法の95%区間は約88.6〜100%であり、未知条件での成功を保証するものではない。',48,y-12,size=9.5)
    y=h2(c,'再現と監査',y-20)
    y=table(c,[['操作','確認するもの'],['python main.py','保存済みモデルで1走行を再実行。'],
        ['python experiments.py','同じ4条件・30seed・4方式を再集計。'],
        ['python verify.py',f"全{audit['missions_audited']}走行の判定ログ、{audit['observations_reinferred']:,}観測を再推論。デモ4本を再走行し照合。"],
        ['pytest -q tests','27項目の制御・入力・保留動作テスト。'],
        ['python train.py','合成データから改良モデルを再学習。']],y,[182,317],size=9.0)
    y=para(c,'評価値・モデル重み・乱数seed・ソース・動画生成コードを同梱。動画の比較例は「旧方式が失敗し改良方式が完走した最初のseed」である63000を使用し、全試行の集計とは区別している。',48,y,size=9.3)
    y=para(c,'参考：scikit-learn「Neural network models (supervised)」「Probability calibration」。URL・実行環境・依存ライブラリはREADME.mdとTHIRD_PARTY_NOTICES.mdに記載。',48,y-15,size=8.5,color=GRAY)
    c.save();print(out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'deliverables/説明資料_PAI最終課題_Toshigakusei.pdf')
    a=p.parse_args();a.out.parent.mkdir(exist_ok=True,parents=True);main(a.out)
