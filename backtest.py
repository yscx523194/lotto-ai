"""
Walk-Forward OOS 백테스트
========================
train_model.py의 전체 파이프라인(XGBoost+Transformer+Scorer+구조예측 → 35개 풀 → 5게임)을
과거 데이터로 반복 검증.

매 회차마다:
1. t회차 이전 데이터로 학습 (retrain_interval마다 재학습)
2. t회차 예측: 풀 35개 + 5게임 생성
3. 실제 당첨번호와 비교

평가 지표:
- 풀 35개에 실제 번호 몇 개 포함? (기대: 4.67, 목표: 5.0+)
- 5게임 중 최고 적중수 (2개 이상 = 가능성)
- 5게임 전체 적중수 합계
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import xgboost as xgb
from scipy import stats

# train_model.py에서 함수 임포트
from train_model import (
    load_data, build_structural_df, detect_regimes,
    extract_structural_features,
    StructuralPredictor, NumberScorer, LottoTransformer,
    train_transformer, predict_transformer,
    score_combination, apply_adjustments, select_pool_35,
    partition_into_5_games, build_pair_scores,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def walk_forward_backtest(train_start=300, retrain_interval=50, verbose=True):
    """
    Walk-Forward 백테스트 (5게임 커버리지 전략)

    Args:
        train_start: 학습 시작 인덱스 (이전 데이터로 첫 학습)
        retrain_interval: 재학습 주기
        verbose: 중간 결과 출력
    """
    t_start = time.time()
    df = load_data()
    total = len(df)
    last_round = int(df.iloc[-1]["round"])

    # 바이너리 매트릭스
    binary_matrix = np.zeros((total, 45), dtype=np.float32)
    for i, row in df.iterrows():
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            binary_matrix[i, int(row[col]) - 1] = 1.0

    struct_df = build_structural_df(df)
    regimes = detect_regimes(struct_df)

    print("=" * 70)
    print(f"  Walk-Forward OOS 백테스트")
    print(f"  범위: {int(df.iloc[train_start]['round'])}회차 ~ {last_round}회차 ({total - train_start}회)")
    print(f"  재학습 간격: {retrain_interval}회")
    print("=" * 70)

    results = []
    xgb_models = None
    tf_model = None
    struct_predictor = StructuralPredictor()
    scorer = NumberScorer()

    for t in range(train_start, total):
        current_regime = max(r for r in regimes if r <= t)
        round_num = int(df.iloc[t]["round"])

        # ── 재학습 ──
        if (t - train_start) % retrain_interval == 0:
            if verbose:
                print(f"\r  [{round_num:>4}회차] 재학습 중...        ", end="", flush=True)

            # XGBoost
            try:
                regime_data = binary_matrix[current_regime:t]
                if len(regime_data) >= 50:
                    X_xgb, y_xgb = [], []
                    for i in range(30, len(regime_data)):
                        X_xgb.append(regime_data[i - 20:i].flatten())
                        y_xgb.append(regime_data[i])
                    X_xgb, y_xgb = np.array(X_xgb), np.array(y_xgb)

                    xgb_models = []
                    for num in range(45):
                        m = xgb.XGBClassifier(
                            n_estimators=100, max_depth=4, learning_rate=0.05,
                            scale_pos_weight=39/6, verbosity=0, random_state=42
                        )
                        m.fit(X_xgb, y_xgb[:, num])
                        xgb_models.append(m)
            except Exception:
                pass

            # Transformer
            try:
                tf_model = train_transformer(binary_matrix, t, seq_len=30, epochs=30)
            except Exception:
                pass

            # Structural Predictor
            try:
                struct_predictor.train(struct_df, t)
            except Exception:
                pass

        # ── 예측 ──
        # XGBoost probs
        xgb_probs = np.ones(45) / 45
        if xgb_models and t >= 20:
            try:
                seq = binary_matrix[t - 20:t].flatten().reshape(1, -1)
                xgb_probs = np.array([m.predict_proba(seq)[0, 1] for m in xgb_models])
            except:
                pass

        # Transformer probs
        tf_probs = predict_transformer(tf_model, binary_matrix, t, seq_len=30)
        if tf_probs is None:
            tf_probs = np.ones(45) / 45

        # Number Scorer
        scorer_probs = scorer.score_numbers(df, t, current_regime)

        # 앙상블 (TF 강화)
        combined = 0.25 * xgb_probs + 0.45 * tf_probs + 0.15 * scorer_probs
        combined = combined / combined.sum()

        # Structural Predictor
        struct_pred = struct_predictor.predict(struct_df, t) if struct_predictor.models else {}

        # 점수 조정 (스트릭 감쇠 제거)
        adjusted, cold_pool = apply_adjustments(combined, df.iloc[:t], t)

        # 풀 35개 (cold 강제 없음)
        pool_35 = select_pool_35(adjusted, cold_pool)

        # 5게임 분할 (빠른 버전)
        if struct_pred:
            # 컨텍스트 정보 계산
            pair_z = build_pair_scores(df, t)
            last_row = df.iloc[t - 1]
            prev_nums = [int(last_row[c]) for c in ["n1", "n2", "n3", "n4", "n5", "n6"]]
            bt_streaks = {}
            for num in range(1, 46):
                streak = 0
                for tt in range(t - 1, -1, -1):
                    row = df.iloc[tt]
                    if num in [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"], row["n6"]]:
                        streak += 1
                    else:
                        break
                bt_streaks[num] = streak

            games = partition_into_5_games(pool_35, struct_pred, pair_z, prev_nums, bt_streaks,
                                           n_attempts=30, n_candidates=100)
        else:
            # struct_pred 없으면 랜덤 파티션
            p = list(pool_35)
            np.random.shuffle(p)
            games = [sorted(p[i*6:(i+1)*6]) for i in range(5)]

        # ── 실제 당첨번호 ──
        actual = set()
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            actual.add(int(df.iloc[t][col]))

        # ── 평가 ──
        pool_hits = len(actual & set(pool_35))
        game_hits = [len(actual & set(g)) for g in games]
        best_game_hits = max(game_hits)
        total_hits = sum(game_hits)

        result = {
            "round": round_num,
            "pool_hits": pool_hits,       # 풀 35개 중 실제 포함 수
            "game_hits": game_hits,       # 각 게임별 적중 수
            "best_game": best_game_hits,  # 최고 적중 게임
            "total_hits": total_hits,     # 5게임 총 적중
            "actual": sorted(actual),
            "pool_35": pool_35,
        }
        results.append(result)

        # 중간 보고
        if verbose and (t - train_start) % 100 == 0 and results:
            recent = results[-min(100, len(results)):]
            avg_pool = np.mean([r["pool_hits"] for r in recent])
            avg_best = np.mean([r["best_game"] for r in recent])
            avg_total = np.mean([r["total_hits"] for r in recent])
            print(f"\r  [{round_num:>4}회차] 풀적중: {avg_pool:.2f}/6  "
                  f"최고게임: {avg_best:.2f}  5게임합: {avg_total:.2f}  "
                  f"({len(results)}회 완료)")

    elapsed = time.time() - t_start

    # ════════════════════════════════════════════════════════════
    # 결과 분석
    # ════════════════════════════════════════════════════════════
    n = len(results)
    pool_hits_all = [r["pool_hits"] for r in results]
    best_game_all = [r["best_game"] for r in results]
    total_hits_all = [r["total_hits"] for r in results]

    print(f"\n{'=' * 70}")
    print(f"  Walk-Forward 백테스트 결과 ({n}회)")
    print(f"{'=' * 70}")

    # 풀 35개 적중률
    print(f"\n  ■ 풀 35개 적중 (랜덤 기대: 4.67/6)")
    print(f"    평균: {np.mean(pool_hits_all):.3f}/6")
    print(f"    분포: ", end="")
    for h in range(7):
        cnt = pool_hits_all.count(h)
        pct = cnt / n * 100
        print(f"{h}적중={cnt}({pct:.1f}%) ", end="")
    print()

    # 5게임 최고 적중
    print(f"\n  ■ 5게임 중 최고 적중 (랜덤 기대: ~1.1)")
    print(f"    평균: {np.mean(best_game_all):.3f}")
    print(f"    분포: ", end="")
    for h in range(7):
        cnt = best_game_all.count(h)
        pct = cnt / n * 100
        if cnt > 0:
            print(f"{h}적중={cnt}({pct:.1f}%) ", end="")
    print()

    # 등수 분석
    prize_counts = {3: 0, 4: 0, 5: 0, 6: 0}
    for r in results:
        bh = r["best_game"]
        if bh >= 3:
            prize_counts[bh] = prize_counts.get(bh, 0) + 1

    print(f"\n  ■ 당첨 횟수 ({n}회 중)")
    print(f"    6개 적중 (1등): {prize_counts.get(6, 0)}회")
    print(f"    5개 적중 (2/3등): {prize_counts.get(5, 0)}회")
    print(f"    4개 적중 (4등): {prize_counts.get(4, 0)}회 ({prize_counts.get(4,0)/n*100:.1f}%)")
    print(f"    3개 적중 (5등): {prize_counts.get(3, 0)}회 ({prize_counts.get(3,0)/n*100:.1f}%)")

    # 랜덤 대비
    # 랜덤 5게임 중 최고 적중 기대값 (시뮬레이션 기반 ~1.1)
    # 랜덤 5게임 4등 확률 ≈ 5 × C(6,4)×C(39,2)/C(45,6) ≈ 5 × 0.0009686 ≈ 0.48%
    random_4plus = 0.00484 * n  # 랜덤 5게임으로 4등+ 기대 횟수
    actual_4plus = prize_counts.get(4, 0) + prize_counts.get(5, 0) + prize_counts.get(6, 0)
    print(f"\n  ■ 랜덤 대비")
    print(f"    풀 적중: {np.mean(pool_hits_all):.3f} vs 랜덤 4.667 "
          f"({'\u2191' if np.mean(pool_hits_all) > 4.667 else '\u2193'} "
          f"{abs(np.mean(pool_hits_all) - 4.667) / 4.667 * 100:.1f}%)")
    print(f"    4등+ 횟수: AI {actual_4plus}회 vs 랜덤 기대 {random_4plus:.1f}회 "
          f"({'↑' if actual_4plus > random_4plus else '↓'} "
          f"{actual_4plus / max(random_4plus, 0.1):.1f}배)")

    # 구간별 분석
    print(f"\n  ■ 구간별 풀 적중 평균")
    chunk = 100
    for i in range(0, n, chunk):
        seg = pool_hits_all[i:i+chunk]
        if seg:
            seg_start = results[i]["round"]
            seg_end = results[min(i+chunk-1, n-1)]["round"]
            print(f"    {seg_start:>4}~{seg_end:>4}회차: "
                  f"풀={np.mean(seg):.3f}  "
                  f"최고게임={np.mean(best_game_all[i:i+chunk]):.3f}  "
                  f"4등+={sum(1 for r in results[i:i+chunk] if r['best_game'] >= 4)}회")

    print(f"\n  소요시간: {elapsed:.0f}초")
    print()

    return results


if __name__ == "__main__":
    # 기본: 301회차부터, 50회마다 재학습
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    results = walk_forward_backtest(
        train_start=start,
        retrain_interval=interval,
        verbose=True,
    )
