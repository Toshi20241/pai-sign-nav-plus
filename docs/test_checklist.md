# 27件の自動チェック：日本語の対応表

用意した入力に対して処理が指定どおりかを点検します。実機の安全性や画像の全問正解を証明するものではありません。

|番号|テスト名|確認すること|
|---:|---|---|
|1|`test_first_view_never_accepted`|Dは1回目の判定だけで進まない。|
|2|`test_agreement`|条件を満たす同じ答えが2回続いたときGOにする。|
|3|`test_disagreement_retries`|1・2回目が違う場合はもう一度確認する。|
|4|`test_third_disagreement_holds`|3回ともそろわない場合はHOLDにする。|
|5|`test_third_consensus_accepts`|2・3回目が一致した場合、その方向を選ぶ。|
|6|`test_low_scores_hold`|3回とも低スコアなら、同方向でも停止する。|
|7|`test_covered_high_score_holds`|一色画像は高スコアでも明暗差不足で停止する。|
|8|`test_reset`|次の標識では以前の確認履歴を使い回さない。|
|9|`test_single_gate`|Cは1回の低スコアで停止する。|
|10|`test_once`|Bは低いスコアでも1回の答えを選ぶ。|
|11|`test_bad_mode`|存在しない方式名をエラーにする。|
|12|`test_features`|AI入力は24×24＝576個の数値になる。|
|13|`test_class_probabilities`|3つの出力スコアの合計が約1になる。正解率の保証ではない。|
|14|`test_blank_contrast`|白一色の画像の明暗差が0になる。|
|15|`test_valid_contrast`|標準的な矢印画像の明暗差が25を超える。|
|16|`test_sensor_determinism[normal]`|同じ条件・同じ位置で同じ画像を再現する。 条件：normal。|
|17|`test_sensor_determinism[variable]`|同じ条件・同じ位置で同じ画像を再現する。 条件：variable。|
|18|`test_sensor_determinism[covered]`|同じ条件・同じ位置で同じ画像を再現する。 条件：covered。|
|19|`test_sensor_determinism[persistent]`|同じ条件・同じ位置で同じ画像を再現する。 条件：persistent。|
|20|`test_capture_pose_guard`|観測位置と向きが不適切な状態での読取りを拒否する。|
|21|`test_geometry_no_labels`|表示用のコース形状データに正解ラベルを入れない。|
|22|`test_covered_visibility`|②矢印なし条件で3位置の明暗差が25未満になる。|
|23|`test_wheel_acceleration`|計算上の車輪加速度が32 rad/s²以下で、上向きへ進む。|
|24|`test_geometry_collision`|柱と重なる地点を接触、開いた地点を非接触と判定する。|
|25|`test_no_goal_credit_for_hold`|①より前の停止を全指示遵守や適切な②前停止に数えない。|
|26|`test_branch_routes_distinct`|左枝と右枝の目標位置がそれぞれ左右にある。|
|27|`test_waypoint_stop`|目標地点にいる場合は前進・旋回指令を0にする。|
