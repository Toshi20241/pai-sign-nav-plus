"""Build a detailed, source-grounded Japanese HTML/PDF explanation (R2).
Browser rendering embeds fonts in the PDF; font files are not distributed.
"""
from __future__ import annotations
import argparse,json,html,subprocess,shutil,base64,re
from pathlib import Path
from vision import ROOT

ASSETS=ROOT/'presentation/assets'

def esc(s):return html.escape(str(s))
def p(s,cls=''):return f'<p class="{cls}">{s}</p>'
def h(s):return f'<h2>{s}</h2>'
def note(s,kind=''):return f'<div class="note {kind}">{s}</div>'
def tbl(headers,rows,widths=None,cls=''):
    cols='' if widths is None else '<colgroup>'+''.join(f'<col style="width:{w}%">' for w in widths)+'</colgroup>'
    return f'<table class="{cls}">'+cols+'<thead><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def image(name,cls='',style=''):return f'<img src="assets/{name}" class="{cls}" style="{style}">'
def figure(img,caption):return '<figure>'+img+f'<figcaption>{caption}</figcaption></figure>'
def samples(names,labels,width='29mm'):
    return '<div class="samples">'+''.join('<figure>'+image(n,style=f'width:{width};height:{width};image-rendering:pixelated')+f'<figcaption>{t}</figcaption></figure>' for n,t in zip(names,labels))+'</div>'

def build_html():
    from report_guard import assert_report_snapshot
    assert_report_snapshot()
    if not (ASSETS/'course.png').exists():
        from make_explanation_assets import main as prepare_assets
        from video_ui import export_figures
        prepare_assets();export_figures()
    audited=json.loads((ROOT/'results/explanation_audit.json').read_text())
    pages=[]
    def page(title,lead,body,source):pages.append((title,lead,body,source))
    page('この作品は、何を試したのか',
         '矢印を読むAIと、ロボットを動かすプログラムを組み合わせ、「一度の判断で進む方式」と「別の位置でも確認してから進む方式」を比べました。',
         '<div class="hero">SIGN NAV PLUS<br><span>矢印を読み、確認してから進む</span></div>'+
         note('<b>実験の舞台は架空です。</b> プログラムで描いた矢印画像、形の決まったコース、平面上の移動計算を使います。実カメラの映像や実機の走行を試したものではありません。')+
         h('3つを分けて読むと、仕組みが分かります')+
         tbl(['区別するもの','この作品での意味'],[
             ('正解','画像を作る前に環境側が決めた、矢印の方向です。採点の基準であり、AIには教えません。'),
             ('AIに見せた画像','回転・ぼかし・ノイズなどを加えて描いた画像です。正解の向きでも、人にもAIにも見えにくいことがあります。'),
             ('AIの判定と選んだ道','AIが画像から方向を答えます。プログラムがその答えを受け、進む・読み直す・止まるを決めます。')],[24,76])+
         h('読む順番')+
         tbl(['ページ','確認できること'],[
             ('2〜4','どの道が正解か。900枚の画像はどこから来たか。見えにくい画像とは何か。'),
             ('5〜7','2つのAIと4つの方式。場所による画像の切替。進行・停止の条件。'),
             ('8〜10','動画の実例を、実際の画像と走行記録で一つずつ確認。'),
             ('11〜14','全試行の結果、言える範囲、再実行の方法、よくある質問。')],[18,82])+
         p('説明の改訂版 R2。認識モデル、移動制御、評価条件、保存済みの480走行の成績は変更せず、用語・図・動画の見せ方を作り直しました。','small'),
         '根拠：vision.py、policy.py、world.py、mission.py。数値は同梱のresults以下の記録に対応。')

    page('どの道を通れば「正しい」のか',
         '3か所の分岐で、標識の指示に従います。どの枝も再び合流するため、ゴールに着くだけでは成功としません。',
         '<div class="two"><div>'+figure(image('course.png',style='height:119mm;width:auto'),
             'コースを上から見た図。下から上へ進む。<br>点線は説明用の正解経路。')+'</div><div>'+
         h('動画の例：実験番号63000')+
         tbl(['標識','正解の指示'],[('①','左 ←'),('②','右 →'),('③','左 ←')],[30,70])+
         p('ここでの「左」は、分岐の手前で上向きに整列したロボットから見た左側です。図でも左側の枝に対応します。「直進 ↑」は中央の枝です。')+
         p('左→右→左は、<b>①では左、次の②では右、③では左</b>という順番です。全部の曲がり角で、その順に曲がるという意味ではありません。')+'</div></div>'+
         note('<b>成功の定義：</b>3つの標識を全部正しく守り、接触せずにゴールへ着くこと。別の道を通って到着した場合は「到着したが、指示違反」と区別します。')+
         h('正解は誰が決めるのか')+
         p('環境プログラムが、各標識を左・直進・右のどれにするかを先に決め、その方向の画像を作ります。AIの判定結果から正解を決め直すことはありません。別の実験番号では標識の組合せも変わります。')+
         p('図や動画に追加した「①左・②右・③左」と点線は、読者のための採点用表示です。AIや進路決定部へ正解の一覧を渡す仕組みではありません。','small'),
         '根拠：world.pyのPracticeWorld.__init__、branch_points、evaluate。実例：results/demo_*/mission.jsonのtrue_labels。')

    page('学習用の900枚は、どこから来たのか',
         'インターネットの写真ではありません。Pythonが矢印を描き、角度・大きさ・位置・明るさなどを変えて作った合成画像です。',
         samples([f'baseline_train_{i}.png' for i in range(3)],['左の学習例','直進の学習例','右の学習例'],'27mm')+
         p('上は基本AIの学習用データを、同じ生成手順で再現した例です。左・直進・右を各300枚用意し、合計900枚にします。コピーを900個並べるのではなく、描画条件を変えます。')+
         tbl(['用途','枚数と内訳','何のためか'],[
             ('基本AIの学習','900枚<br>各方向300枚','弱い加工の画像で、矢印の分類を学習。'),
             ('強化AIの学習','4,500枚<br>各方向1,500枚','弱い加工と強い加工を混ぜて学習。'),
             ('強化AIの途中確認','600枚<br>各方向200枚','学習中の成績を見るための、別の画像。'),
             ('弱い加工の画像テスト','900枚<br>各方向300枚','学習が終わった後、同じ画像を両AIに出題。'),
             ('強い加工の画像テスト','900枚<br>各方向300枚','見えにくい画像で、両AIの正解枚数を比較。')],[24,27,49])+
         note('<b>900枚という数字が2種類あります。</b><br>「基本AIを学習させた900枚」と「強い加工のテスト900枚」は別のデータです。テスト画像を使って学習したわけではありません。')+
         h('「学習」「確認」「テスト」の違い')+
         p('学習では、画像と正解の方向を使ってAIの内部の数値を調整します。途中確認は学習状況を見るためです。テストでは内部の数値を変えず、まだ学習に使っていない画像にどれだけ正解するかを数えます。')+
         p('生成番号は、基本学習11001、強化学習51001、途中確認52001、弱い加工テスト53001、強い加工テスト54001です。番号は同じ画像群を作り直すための乱数seedです。','small'),
         '根拠：train_baseline.py、train.pyのdataset／SPLITS、vision.pyのrender_sign。再確認：results/explanation_audit.json。')

    page('「強い画像変動900枚」を具体的に見る',
         'この表現は「強い加工を加えた、テスト用の矢印画像が900枚」という意味です。900回のロボット走行ではありません。',
         '<div class="sample-label">弱い加工のテスト画像の例</div>'+
         samples([f'test_standard_{i}.png' for i in range(3)],['正解：左','正解：直進','正解：右'],'25mm')+
         '<div class="sample-label">強い加工のテスト画像の例</div>'+
         samples([f'test_stress_{i}.png' for i in range(3)],['正解：左','正解：直進','正解：右'],'25mm')+
         p('正解の方向を決めてから加工します。強く加工した画像には、人が見ても方向を読み取りにくいものがあります。上の例は各テスト集合で各方向が最初に現れた画像で、見栄えによる選別はしていません。','small')+
         tbl(['変えるもの','弱い加工','強い加工'],[
             ('回転角度','−13〜13度','−38〜38度'),('大きさの倍率','0.78〜1.12倍','0.52〜0.95倍'),
             ('中心位置のずれ','左右・上下に最大5画素','左右・上下に最大11画素'),
             ('ぼかし／ノイズ','弱い範囲','より強い範囲'),('一部を隠す加工','なし','45％の確率で横長の帯を描く')],[30,35,35])+
         note('この加工範囲は実験の設定です。雨・暗所・遠距離撮影など、実際の環境の厳しさをそのまま再現・測定した数値ではありません。')+
         p('画像は96×96画素。学習時も評価時も24×24画素の白黒へ縮小してAIに入力します。元画像をそのまま表示し、図では見やすい大きさに拡大しています。','small'),
         '根拠：vision.pyのrender_sign／preprocess。表示画像：test_standard(seed53001)、test_stress(seed54001)から再生成。')

    page('2つの画像AIと、4つの走り方を区別する',
         '「モデル」は画像を分類するAIのことです。「方式」は、そのAIと進行・停止のルールを組み合わせたものです。',
         h('画像AIは2種類')+
         tbl(['名称','学習画像','AIの構成'],[
             ('基本AI<br>旧版の「旧モデル」','900枚・弱い加工','576 → 32 → 3'),
             ('強化AI<br>旧版の「改良モデル」','4,500枚・加工の強弱を混合','576 → 64 → 32 → 3')],[28,37,35])+
         p('どちらもMLPという画像分類用のニューラルネットワークです。24×24＝576個の明るさの数値を入力し、最後に左・直進・右の3つのスコアを出します。32、64は途中の計算に使うユニット数です。')+
         note('基本AIと強化AIでは、<b>学習画像、ネットワークの大きさ、学習回数などをまとめて変更</b>しています。性能の違いを「画像枚数だけの効果」や「層を増やした効果だけ」とは説明しません。')+
         h('走り方は4方式')+
         tbl(['記号','画像AI','判定した後の動き'],[
             ('A','基本AI','1回だけ読み、最もスコアが高い方向へ進む。'),
             ('B','強化AI','Aと同じく1回だけ読み、その方向へ進む。'),
             ('C','強化AI','1回読み、スコアと明暗差の条件を満たせば進む。満たさなければ止まる。'),
             ('D','強化AI','別位置でも読み、条件を満たした判定が直近2回一致したら進む。最大3回でも確認できなければ止まる。')],[9,18,73])+
         p('<b>AとB</b>は、画像AIをまとめて強化した効果の比較です。<b>BとD</b>は同じ強化AIを使い、再確認を含む判断ルール全体の違いを比べます。<b>CとD</b>は、止めるだけでなく別位置で読み直す価値を見る比較です。')+
         p('「旧／改良」だけではAIと走り方の変更が混ざるため、以降は基本AI／強化AI、方式A〜Dという名称で統一します。','small'),
         '根拠：policy.pyのPOLICIES／DecisionPolicy、train_baseline.py、train.py。')

    page('位置を変えると、なぜ画像が見やすくなるのか',
         'この実験では、位置ごとに画像の生成条件を決めています。カメラを動かした結果を、光学的に計算したものではありません。',
         samples([f'demo_active_j3_v{v}.png' for v in (1,2,3)],['③・位置1：強い加工','③・位置2：弱い加工','③・位置3：弱い加工'],'31mm')+
         p('上は動画の③で、実際にAIへ渡した3枚です。ロボットは分岐の手前の3つの既知の観測位置へ移動し、停止して上向きに整列します。その位置番号に対応する合成画像を受け取ります。')+
         tbl(['実験条件','位置1','位置2・3','確かめること'],[
             ('通常','弱い加工','弱い加工','読みやすい条件で指示を守るか。'),
             ('位置変更で改善','強い加工','弱い加工','より見やすい入力を得られる場合、再確認が役立つか。'),
             ('見えにくさが持続','強い加工','強い加工','移動しても読みにくい場合の限界。'),
             ('②の矢印なし','②は矢印を描かない','②は同じく描かない','推測で②へ進まずに止まるか。①・③は「位置変更で改善」と同じ。')],[23,18,21,38])+
         note('<b>「位置変更で改善」は、再確認で情報を増やせるようにした条件です。</b>この結果だけで「実際のロボットも移動すれば必ず見える」とは言えません。そこで「見えにくさが持続」する条件も別に試しました。')+
         h('同じ画像で比較する、とは')+
         p('実験番号（seed）・標識番号・位置番号・画像条件が同じなら、方式A〜Dは同じ画像を受け取ります。AIの答えを見て、画像を易しくする処理はありません。A〜Cが1回で決めるのに対し、Dだけが別位置の追加画像を使います。これは意図した方式の違いです。'),
         '根拠：world.pyのstations／_panel／capture_sign。画像：results/demo_active/camera/j3_v1〜3.png。')

    page('方式Dは、何を確認してから進むのか',
         '「同じ方向を2回言った」だけでは進みません。2回ともスコアと明暗差の条件を満たし、直近2回の方向が一致したときに進みます。',
         '<div class="flow">位置1で読取り → 必要なら位置2へ → 必要なら位置3へ<br><b>条件を満たす2回の一致：進む ／ 最大3回でも未確認：止まる</b></div>'+
         tbl(['確認項目','条件','何を意味するか'],[
             ('AIの出力スコア','0.95以上','AIが選んだ方向のスコア。95％正解の保証ではない。'),
             ('画像の明暗差','25以上','矢印のないほぼ一色の画像を除くための簡単な検査。'),
             ('直近2回の一致','同じ方向','2回とも上の2条件を満たし、同じ方向を答える。'),
             ('観測回数の上限','3回','条件がそろわないまま確認を無限に続けない。')],[25,20,55])+
         h('③の実際の例')+
         tbl(['観測','AI判定','スコア','扱い'],[
             ('1回目','左','0.743','0.95未満なので、確定に使わない。'),
             ('2回目','左','約1.000','条件を満たす1回目。まだ確定しない。'),
             ('3回目','左','約1.000','2・3回目が両方条件を満たして一致。左へ進む。')],[16,14,18,52])+
         p('1回目の答えが正しくても、方式Dは正解ラベルを知らないため「もう正解だから進もう」とは判断できません。決めた条件で確認を続けます。')+
         h('数値設定の位置付け')+
         p('0.95、25、2回一致、最大3回は今回の固定設定です。最適値だと証明した数値ではありません。別番号の入力でスコア閾値を変えた確認記録もありますが、480走行では設定を変えていません。')+
         p('明暗差は、画像中央の白黒画素で「明るさを暗い順に並べたときの95％点−5％点」を計算した値です。文字や模様のある未知画像を一般的に見分ける仕組みではありません。','small'),
         '根拠：policy.pyのGateConfig／observe、vision.pyのreadability。③の記録：demo_active/mission.json。')

    page('動画の道が違う理由を、最後まで追う',
         '同じ実験番号63000、同じ標識「左・右・左」です。方式Aは③で直進を選び、方式Dは左を選びました。',
         '<div class="maps3">'+''.join('<figure>'+image('route_'+n+'.png',style='height:92mm;width:auto')+f'<figcaption><b>{t}</b></figcaption></figure>' for n,t in [('A','A：基本AI・1回'),('B','B：強化AI・1回'),('D','D：強化AI・再確認')])+'</div>'+
         tbl(['標識','正解','Aの選択','Bの選択','Dの選択'],[
             ('①','左','左 ○','左 ○','左 ○'),('②','右','右 ○','右 ○','右 ○'),('③','左','直進 ×','左 ○','左 ○'),
             ('ゴール到着','—','到着','到着','到着'),('指示を全部守ったか','—','いいえ','はい','はい')],[14,14,24,24,24])+
         note('Aは「到着したが、③で指示違反」です。Dは「3つとも守って到着」です。<b>同じ強化AIを使うBも、この1例では正解しています。</b>したがって、この動画例だけで再確認の必要性や効果を結論付けません。')+
         p('この例は、保存済み30試行のうちAが失敗しDが成功した最初の実験番号として選ばれています。成績の集計は、選んだ1例ではなく30試行すべてを使っています。','small'),
         '根拠：results/demo_baseline、demo_once、demo_activeのmission.json／telemetry.csv。例の選択：results/demo_selection.json。')

    page('③の「正しい画像」は、どれなのか',
         '正解は「左」という方向です。後から得た見やすい画像が、正解を決めるのではありません。見本・入力画像・AIの答えを分けて示します。',
         '<div class="two"><div>'+figure(image('reference_0.png',style='width:30mm;height:30mm'),'<b>正解を説明する見本：左</b><br>この見本はAIへ渡していない。')+'</div><div>'+figure(image('demo_active_j3_v1.png',style='width:30mm;height:30mm;image-rendering:pixelated'),'<b>位置1の実際の入力画像</b><br>A・B・Dに同じ画像を渡した。')+'</div></div>'+
         tbl(['方式','位置1でのAI判定','その直後の動き'],[
             ('A 基本AI・1回','直進／スコア0.889','直進の枝を選ぶ。正解の左と違う。'),
             ('B 強化AI・1回','左／スコア0.743','左の枝を選ぶ。この例では正解。'),
             ('D 強化AI・再確認','左／スコア0.743','閾値0.95未満なので、位置2へ移動して確認を続ける。')],[26,31,43])+
         samples(['demo_active_j3_v2.png','demo_active_j3_v3.png'],['Dが位置2で得た入力画像','Dが位置3で得た入力画像'],'28mm')+
         p('Dは位置2・3の両方で「左」と判定し、スコアと明暗差の条件も満たしました。この2回の一致で左の枝を確定します。2・3枚目は、Dだけが移動して得た追加の入力画像です。')+
         note('<b>この例で言えること：</b>基本AIは最初の画像を読み違えました。強化AIは最初から左と答えました。Dはさらに確認してから動きました。<b>「再確認したから初めて左と分かった」例ではありません。</b>')+
         p('動画の左右は同じ時刻で再生していますが、Dは確認に時間を使うため、左右で読んでいる標識が違う瞬間があります。新しい動画では標識番号と位置番号を明示しています。','small'),
         '根拠：各demoのcamera/j3_v1.pngの画素ハッシュ一致、mission.jsonのprediction／action。追加画像：demo_active/camera/j3_v2〜3.png。')

    page('②の矢印が、何度見てもない場合',
         '停止を試すため、②ではどの観測位置でも矢印を描かない条件を作りました。実物の板で遮った映像ではなく、画像生成上の設定です。',
         samples([f'demo_hold_j2_v{v}.png' for v in (1,2,3)],['②・位置1：矢印なし','②・位置2：矢印なし','②・位置3：矢印なし'],'29mm')+
         tbl(['観測','AIの答え','スコア','明暗差','方式Dの行動'],[
             ('1回目','左','0.776','7','位置2へ移動'),('2回目','左','0.704','7','位置3へ移動'),('3回目','左','0.800','7','停止を確定')],[14,14,17,14,41])+
         note('矢印がなくても、このAIは3択のどれかを出力します。<b>「左」と出たことを、読めた証拠とは扱いません。</b>この例ではスコアも明暗差も条件未達で、3回確認して停止します。')+
         '<div class="two"><div>'+figure(image('route_hold.png',style='height:88mm;width:auto'),'①は左を選び、②の手前で停止。')+'</div><div>'+h('何を成功として数えるか')+
         p('停止試験の成功は、①の指示を守り、読めない②の分岐へ入る前で止まることです。①より前に止まった場合は、この停止試験の成功に数えません。')+
         p('方式Dは30試行すべてで、この位置で停止しました。ゴール到着は0件です。停止の30件を「完走30件」として加算していません。')+
         p('プログラム内には②の元の方向が残っていても、矢印なし条件ではその方向を当てたかどうかを採点しません。進入したか、適切な位置で止まったかを評価します。','small')+'</div></div>',
         '根拠：world.pyのcovered条件／evaluate、results/demo_hold/mission.json、benchmark/summary.json。')

    page('全体の結果：画像の枚数と、走行の回数を分ける',
         '画像分類の試験とロボット走行の試験は別です。前者は正解した画像の枚数、後者は指示を守った走行の回数を数えます。',
         h('試験1：同じ900枚の画像を、2つのAIに出題')+
         tbl(['テスト画像','基本AI','強化AI'],[
             ('弱い加工・900枚','900 / 900枚（100.0％）','900 / 900枚（100.0％）'),
             ('強い加工・900枚','421 / 900枚（46.8％）','798 / 900枚（88.7％）')],[34,33,33])+
         p('画像試験ではロボットを走らせません。900枚は各方向300枚で、両AIに同じ画像を渡しています。')+
         h('試験2：指示どおりに到着した回数（各欄30走行）')+
         tbl(['方式','通常','位置変更で改善','見えにくさが持続'],[
             ('A 基本AI・1回','30 / 30','2 / 30','2 / 30'),('B 強化AI・1回','30 / 30','18 / 30','18 / 30'),
             ('C 強化AI・条件停止','30 / 30','6 / 30','6 / 30'),('D 強化AI・再確認','30 / 30','30 / 30','6 / 30')],[34,19,24,23])+
         p('「位置変更で改善」では、同じ強化AIのBが18回、Dが30回成功しました。Cは23回停止し、残る7回が到着しましたが、そのうち1回は指示違反です。このため成功は6回です。','small')+
         h('試験3：②の矢印なし（各欄30走行）')+
         tbl(['方式','②の手前で適切に停止','読めない②へ進入'],[
             ('A 基本AI・1回','0 / 30','30 / 30'),('B 強化AI・1回','0 / 30','30 / 30'),
             ('C 強化AI・条件停止','18 / 30','0 / 30'),('D 強化AI・再確認','30 / 30','0 / 30')],[34,33,33])+
         p('Cの残り12回は①より前で停止しました。走行試験の総数は4条件×30実験番号×4方式＝480回です。試験3もこの480回に含みます。','small')+
         note('見えにくさが持続する条件で、Dは6回到着、24回停止でした。Bの18回到着より少なく、再確認が常に到着数を増やすわけではありません。'),
         '根拠：results/vision_metrics.json、test_standard.csv、test_stress.csv、benchmark/trials.json／summary.json。')

    page('この結果から言えること・言えないこと',
         '「再確認すれば必ずうまくいく」という結論ではありません。見やすい追加画像を得られる条件では有効で、得られなければ停止が増えます。',
         h('良くなったことと、追加で必要になった時間')+
         p('位置変更で見やすくなる条件では、DはBより指示どおりに到着する回数が増えました。矢印なし条件では、Dは全30回で適切な位置に停止しました。')+
         tbl(['通常条件の平均走行時間','結果'],[
             ('1回で決める方式A・B','66.9秒'),('再確認する方式D','79.2秒'),('確認のための追加時間','約12.3秒')],[65,35])+
         p('この比較は同じ30実験番号の通常条件です。読取り・向きの調整・位置変更を増やす分、時間がかかります。動画例1本の所要時間との差とは別の数字です。')+
         h('条件と限界')+
         tbl(['今回できていること','今回まだ確かめていないこと'],[
             ('合成画像の生成・分類・判定ルールの比較','実カメラでの標識検出、遠近投影、実画像での認識性能。'),
             ('既知の地点への移動と、車輪速度の計算','未知のコース探索、自己位置の推定、未知障害物への回避。'),
             ('滑りなしの平面運動、車輪の遅れと加速度制限','実機の摩擦・滑り・接触を含む詳細な物理挙動。'),
             ('固定条件の30試行、失敗も含めた集計','未知条件での100％成功や、実環境の安全性。')],[48,52])+
         h('比較の読み方')+
         p('A対Bはモデルの構成・学習データ・学習設定をまとめた変更です。B対Dは同じ画像AIですが、追加観測と確認ルールの両方が変わります。個々の要素の寄与をすべて分解した実験ではありません。')+
         p('接触0件は、この既知経路と円・矩形の重なり判定での観測です。出力スコア0.95は正解率95％の保証ではなく、2回一致しても同じ誤りを繰り返す可能性は残ります。','small'),
         '根拠：drive.py、world.py、policy.py、benchmark/summary.json。誤読を防ぐための解釈範囲を明示。')

    page('コードのどこで、何をしているのか',
         '画像を作る部分、画像を分類する部分、進むか止まるかを決める部分、車輪を動かす部分を分けています。',
         tbl(['ファイル','役割','確認する入口'],[
             ('vision.py','矢印を描く。24×24へ縮小する。学習済みAIで分類する。','render_sign / preprocess / SignClassifier'),
             ('train_baseline.py<br>train.py','合成画像と正解の組で、基本AI／強化AIを学習する。','dataset / main'),
             ('world.py','コース、観測位置、位置ごとの入力画像、採点基準。','PracticeWorld'),
             ('policy.py','AIの出力と明暗差から、進行・再確認・停止を決める。','DecisionPolicy.observe'),
             ('mission.py','位置へ移動→整列→観測→分岐走行の順序を実行する。','run_episode'),
             ('drive.py','目標地点へ向かう速度を計算し、左右輪の運動を進める。','DifferentialDrive'),
             ('experiments.py<br>verify.py','全条件を試し、入力画像・判断・集計が一致するか照合する。','main')],[25,47,28],'smalltable')+
         h('まず1回、保存済みモデルで動かす')+
         '<pre>python -m pip install -r requirements-core.txt\npython main.py</pre>'+
         p('画面に数値の結果が出て、results/demo_active/へ画像・判断・走行記録を保存します。動画やゲーム画面が自動で開くコマンドではありません。')+
         h('結果を確認する')+
         '<pre>python verify.py\npython -m pip install -r requirements.txt\npytest -q tests</pre>'+
         p('この改訂作業では、480走行の記録と1,673回の画像判断を再照合し、デモ4本を再走行しました。元の27項目のテストも通りました。画像テストは900枚ずつ再生成し、正解数と保存済み判定の一致を確認しています。')+
         p('これは制作環境での確認記録です。利用者ご自身のMacやColabで実行済みという意味ではありません。操作の詳細と依存ライブラリはREADME.mdに記載しています。','small'),
         '根拠：results/verification.json、unit_tests.txt、explanation_audit.json。コアコードと学習済みモデルは改訂前から変更なし。')

    page('質問されたときに説明する要点',
         '「何を作ったか」「何をあらかじめ設定したか」「何を比較したか」を分けて答えると、実験の範囲が伝わります。',
         h('Q. なぜ基本AIと強化AIで道が違うのですか？')+
         p('動画例では基本AIが③の左矢印を直進と読み違えたからです。強化AIは左と答えています。どの枝も合流するため、基本AIもゴールに到着しますが、指示違反です。')+
         h('Q. 読み直すと見やすくなるのは、作られた設定ですか？')+
         p('はい。位置1は強い加工、位置2・3は弱い加工という設定です。その条件下で再確認の動作を試しています。全位置で強い加工が続く場合も試し、停止が増えることを確認しています。')+
         h('Q. この動画だけで、再確認が有効だと分かりますか？')+
         p('この1例では、再確認のない強化AIの方式Bも成功しています。再確認を含む方式の効果は、同じ強化AIのBとDを各30走行で比較して判断します。')+
         h('Q. なぜ2回一致、最大3回にしたのですか？')+
         p('1回だけで決めず、確認を無限に続けない構成を試すための固定設定です。最適回数を求める実験ではありません。誤った方向を2回続けて答える可能性は残ります。')+
         h('Q. 何がAIで、何が普通のプログラムですか？')+
         p('AIは画像を3方向に分類するMLPです。再確認・停止の条件、コースの経由点、車輪の速度計算は通常のプログラムです。ロボットの走り方そのものを学習させた作品ではありません。')+
         note('<b>説明のまとめ：</b><br>合成矢印を読むAIと移動制御をつなぎ、1回で進む・条件未達なら止まる・別位置で確認する、という方式を比較しました。正しい到着と停止を分け、効果だけでなく時間と適用範囲も記録しました。')+
         p('<b>用語：</b>モデル＝学習済みの画像分類AI。観測＝決めた位置で画像を受け取ること。seed＝同じ条件を再現するための乱数の初期値。判断保留＝方向を確定せず、分岐の手前で停止すること。','small'),
         '根拠：前各ページのコード・記録。説明資料は実装済みの内容と設定上の前提に限定。')

    from report_content_r3 import revise_pages
    pages=revise_pages(pages)

    css='''
@page { size:A4; margin:0; }
*{box-sizing:border-box}html,body{margin:0;padding:0;background:#fff;color:#183046;font-family:"Noto Sans CJK JP","Noto Sans CJK",sans-serif;font-size:10.8pt;line-height:1.57}
.page{width:210mm;height:297mm;position:relative;padding:13mm 16mm 15mm;break-after:page;page-break-after:always;overflow:hidden}
.page:last-child{break-after:auto;page-break-after:auto}.head{display:flex;justify-content:space-between;font-size:8pt;color:#007d80;border-bottom:.3mm solid #d3dce5;padding-bottom:3mm;margin-bottom:5mm}
.kicker{font-size:9pt;font-weight:700;color:#007d80;margin-bottom:2mm}h1{font-size:21pt;line-height:1.4;margin:0 0 3mm;font-weight:700;letter-spacing:-.2pt}h2{font-size:12.3pt;line-height:1.45;border-left:1.1mm solid #007d80;padding-left:2.5mm;margin:4mm 0 2mm;color:#163047}
p{margin:2.3mm 0}p.lead{font-size:11.3pt;font-weight:500;line-height:1.6;margin-bottom:4mm}.small{font-size:9.3pt;color:#435d6e;line-height:1.55}.note{background:#eaf5f4;border-left:1mm solid #007d80;padding:3mm 3.5mm;margin:3.3mm 0;line-height:1.58}.note.warn{background:#fff1e9;border-color:#b94f17}
table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:9.8pt;line-height:1.5;margin:2mm 0 3mm}th{background:#183046;color:#fff;font-weight:600;text-align:left;padding:2.1mm 2mm}td{vertical-align:top;border-bottom:.23mm solid #dae2e8;padding:2.1mm 2mm;word-break:normal;overflow-wrap:anywhere}tr:nth-child(even) td{background:#f2f6f9}.smalltable{font-size:9.1pt}
.samples{display:flex;justify-content:space-evenly;align-items:flex-start;gap:4mm;margin:2.5mm 0}.samples figure{flex:1;text-align:center;margin:0}.samples img{border:.25mm solid #c6d1dc}.sample-label{font-size:10pt;font-weight:bold;color:#007d80;margin-top:2mm}
figure{margin:2mm 0;text-align:center}figcaption{font-size:8.7pt;line-height:1.45;color:#40586b;margin-top:1.2mm}img{max-width:100%}.two{display:grid;grid-template-columns:1fr 1.35fr;gap:6mm;align-items:start}.two>div:first-child{min-width:0}.two h2{margin-top:2mm}.maps3{display:flex;justify-content:space-around;gap:3mm}.maps3 figure{width:32%;margin:0}
.hero{background:#183046;color:#fff;padding:6mm 5mm;margin:3mm 0 4mm;font-size:26pt;font-weight:bold;line-height:1.5}.hero span{font-size:18pt}.flow{background:#edf2f8;padding:4mm;text-align:center;font-size:11.2pt;line-height:1.9;margin:3mm 0}pre{font-family:"DejaVu Sans Mono",monospace;white-space:pre-wrap;word-break:break-all;background:#edf2f8;padding:3mm;font-size:9.2pt;line-height:1.6;margin:2mm 0}
.source{position:absolute;left:16mm;right:16mm;bottom:15mm;font-size:7.3pt;line-height:1.45;color:#5b6d7c;border-top:.2mm solid #d8e1e8;padding-top:1.7mm}.foot{position:absolute;bottom:6.5mm;left:16mm;right:16mm;display:flex;justify-content:space-between;font-size:7.8pt;color:#5b6d7c}.body{height:245mm} #p12 td{padding:1.7mm 2mm} #p12 h2{margin-top:3mm} #p18 pre{padding:2mm 3mm} #p18 .smalltable td{padding:1.8mm 2mm}
'''
    s='<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>SIGN NAV PLUS 共通提出版 R3</title><style>'+css+'</style></head><body>'
    for i,(title,lead,body,source) in enumerate(pages,1):
        s+=f'<section class="page" id="p{i}"><div class="head"><span>SIGN NAV PLUS / 共通提出版 R3</span><span>Toshigakusei</span></div><div class="body"><div class="kicker">{i:02d} / {len(pages):02d}</div><h1>{title}</h1>'+p(lead,'lead')+body+'</div><div class="source">'+source+f'</div><div class="foot"><span>合成標識・既知コース・平面移動の学習用シミュレーション</span><span>{i} / {len(pages)}</span></div></section>'
    s+='</body></html>'
    def embed_image(m):
        blob=(ASSETS/m.group(1)).read_bytes()
        return 'src="data:image/png;base64,'+base64.b64encode(blob).decode()+'"'
    s=re.sub(r'src="assets/([^"]+)"',embed_image,s)
    dest=ROOT/'presentation/report.html';dest.write_text(s,encoding='utf-8');return dest,len(pages)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'deliverables/説明資料_PAI最終課題_Toshigakusei.pdf');parser.add_argument('--html-only',action='store_true');args=parser.parse_args()
    path,n=build_html()
    if args.html_only:print(path);return
    from playwright.sync_api import sync_playwright
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=shutil.which('chromium') or None,args=['--no-sandbox'])
        page=browser.new_page()
        content=path.read_text(encoding='utf-8')
        def embed(m):
            asset=(path.parent/m.group(1)).read_bytes()
            return 'src="data:image/png;base64,'+base64.b64encode(asset).decode()+'"'
        content=re.sub(r'src="(assets/[^"]+)"',embed,content)
        page.set_content(content,wait_until='networkidle')
        page.emulate_media(media='print')
        metrics=page.evaluate('''() => Array.from(document.querySelectorAll('.page')).map(el=>{let src=el.querySelector('.source').getBoundingClientRect();let body=el.querySelector('.body');let children=Array.from(body.children);let bottom=Math.max(...children.map(x=>x.getBoundingClientRect().bottom));return {page:el.id,contentBottom:bottom-el.getBoundingClientRect().top,sourceTop:src.top-el.getBoundingClientRect().top,overlap:bottom>src.top-8};})''')
        (ROOT/'results/pdf_layout_check.json').write_text(json.dumps(metrics,indent=2))
        bad=[m for m in metrics if m['overlap']]
        if bad:
            browser.close();raise RuntimeError('Text overlaps source footer: '+str(bad))
        page.pdf(path=str(args.out),format='A4',print_background=True,prefer_css_page_size=True)
        browser.close()
    print(f'{n} pages: {args.out}')
if __name__=='__main__':main()
