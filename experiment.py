"""
로또 AI 모델 체계적 실험 프레임워크
===================================
다양한 하이퍼파라미터 / 필터 / 앙상블 가중치를 자동으로 테스트하고
최적 조합을 찾는다.

실험 대상:
  A) 필터 강도 (하드/소프트 필터 On/Off)
  B) 앙상블 가중치 (XGB / Transformer / Scorer 비율)
  C) XGBoost 하이퍼파라미터
  D) Transformer 아키텍처
  E) 풀 크기 (25 / 30 / 35)
  F) Cold enforcement 비율
  G) Partition 최적화 강도
"""

import os, json, time, copy
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

from train_model import (
    load_data, build_structural_df, detect_regimes,
    extract_structural_features,
    StructuralPredictor, NumberScorer, LottoTransformer,
    train_transformer, predict_transformer,
    build_pair_scores,
)

# ════════════════════════════════════════════════════════════
# 파라미터화된 score_combination
# ════════════════════════════════════════════════════════════

def score_combination_v(combo, struct_pred, cfg, pair_z=None, prev_nums=None, streaks=None):
    """설정 기반 조합 점수 — cfg dict로 필터/가중치 조절"""
    nums = sorted(combo)
    total_sum = sum(nums)
    odd_count = sum(1 for n in nums if n % 2 == 1)
    consec_count = sum(1 for i in range(5) if nums[i + 1] - nums[i] == 1)
    high_count = sum(1 for n in nums if n >= 23)
    has_triple = any(nums[i+1] - nums[i] == 1 and nums[i+2] - nums[i+1] == 1 for i in range(4))
    gaps = [nums[i+1] - nums[i] for i in range(5)]

    diffs = set()
    for i in range(6):
        for j in range(i + 1, 6):
            diffs.add(abs(nums[j] - nums[i]))
    ac_value = len(diffs) - 5

    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
    prime_count = sum(1 for n in nums if n in primes)
    endings = [n % 10 for n in nums]
    max_same_ending = max(Counter(endings).values())
    unique_endings = len(set(endings))
    ending_sum = sum(endings)

    decades = [0] * 5
    for n in nums:
        if n <= 9: decades[0] += 1
        elif n <= 19: decades[1] += 1
        elif n <= 29: decades[2] += 1
        elif n <= 39: decades[3] += 1
        else: decades[4] += 1

    span = nums[5] - nums[0]
    has_quad = any(nums[i+1]-nums[i]==1 and nums[i+2]-nums[i+1]==1 and nums[i+3]-nums[i+2]==1 for i in range(3))
    mult3_count = sum(1 for n in nums if n % 3 == 0)
    mult5_count = sum(1 for n in nums if n % 5 == 0)
    max_decade = max(decades)
    min_gap = min(gaps)
    gap_cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0

    # ── 하드 필터 (항상 적용 — 출현율 < 5%) ──
    if total_sum < 90 or total_sum > 200: return -100.0
    if odd_count <= 0 or odd_count >= 6: return -100.0
    if ac_value <= 4: return -100.0
    if max_same_ending >= 4: return -100.0
    if prime_count >= 5: return -100.0
    if has_quad: return -100.0
    if span < 20: return -100.0
    if high_count == 0 or high_count == 6: return -100.0
    if nums[0] >= 15: return -100.0
    if nums[5] <= 35: return -100.0

    # 선택적 하드 필터
    if cfg.get("hard_decade5", True) and max_decade >= 5: return -100.0
    if cfg.get("hard_mingap5", True) and min_gap >= 5: return -100.0
    if cfg.get("hard_endsum10", True) and ending_sum <= 10: return -100.0
    if cfg.get("hard_carry4", True) and prev_nums is not None:
        if len(set(nums) & set(prev_nums)) >= 4: return -100.0
    if cfg.get("hard_streak2", True) and streaks is not None:
        if sum(1 for n in nums if streaks.get(n, 0) >= 4) >= 2: return -100.0

    # ── 소프트 필터 ──
    log_prob = 0.0
    sw = cfg.get("soft_weight", 1.0)  # 소프트 필터 전체 강도

    if has_triple: log_prob -= 1.5 * sw
    if odd_count == 1 or odd_count == 5: log_prob -= 0.5 * sw
    if consec_count >= 3: log_prob -= 1.0 * sw
    if max_decade >= 4: log_prob -= 1.0 * sw
    if unique_endings <= 3: log_prob -= 1.0 * sw
    if mult3_count == 0 or mult3_count >= 5: log_prob -= 0.5 * sw
    if total_sum < 100 or total_sum > 170: log_prob -= 0.3 * sw

    if cfg.get("soft_extra", True):
        if mult5_count >= 4: log_prob -= 0.8 * sw
        if gap_cv < 0.3: log_prob -= 0.8 * sw
        if ending_sum >= 40: log_prob -= 0.5 * sw
        if prev_nums is not None and len(set(nums) & set(prev_nums)) >= 3:
            log_prob -= 0.8 * sw
        if streaks is not None:
            s3 = sum(1 for n in nums if streaks.get(n, 0) >= 3)
            if s3 >= 2: log_prob -= 0.5 * sw
            if s3 >= 3: log_prob -= 1.0 * sw

    # ── 구조 예측 z-score ──
    struct_w = cfg.get("struct_weight", 0.5)
    checks = {
        "sum": total_sum, "odd_count": odd_count, "high_count": high_count,
        "consec_count": consec_count, "ac_value": ac_value,
        "prime_count": prime_count, "unique_endings": unique_endings,
        "decade_0": decades[0], "decade_1": decades[1], "decade_2": decades[2],
        "decade_3": decades[3], "decade_4": decades[4],
    }
    for key, actual_val in checks.items():
        if key in struct_pred:
            pred = struct_pred[key]["pred"]
            std = struct_pred[key]["std"]
            z = (actual_val - pred) / std
            log_prob -= struct_w * z * z

    # ── 페어 점수 ──
    pair_w = cfg.get("pair_weight", 0.05)
    if pair_z is not None and pair_w > 0:
        pair_score = 0.0
        count = 0
        idxs = [n - 1 for n in nums]
        for i in range(6):
            for j in range(i + 1, 6):
                pair_score += pair_z[idxs[i], idxs[j]]
                count += 1
        log_prob += pair_w * pair_score / count

    return log_prob


# ════════════════════════════════════════════════════════════
# 파라미터화된 파티션
# ════════════════════════════════════════════════════════════

def partition_v(pool_30, struct_pred, cfg, pair_z=None, prev_nums=None, streaks=None):
    pool = list(pool_30)
    n_attempts = cfg.get("n_attempts", 30)
    n_candidates = cfg.get("n_candidates", 100)
    best_partition = None
    best_total_score = -999

    for _ in range(n_attempts):
        partition = []
        remaining = list(pool)
        np.random.shuffle(remaining)
        # 풀이 30개 초과면 30개만 사용 (5게임×6=30)
        if len(remaining) > 30:
            remaining = remaining[:30]
        total_score = 0

        for game_idx in range(5):
            if len(remaining) < 6:
                break
            if game_idx == 4:
                combo = sorted(remaining[:6])
                partition.append(combo)
                total_score += score_combination_v(tuple(combo), struct_pred, cfg, pair_z, prev_nums, streaks)
                remaining = []
            else:
                best_combo, best_sc = None, -999
                if len(remaining) <= 18:
                    candidates = list(combinations(remaining, 6))
                    np.random.shuffle(candidates)
                    candidates = candidates[:n_candidates]
                else:
                    candidates = set()
                    for _ in range(n_candidates):
                        idx = np.random.choice(len(remaining), 6, replace=False)
                        candidates.add(tuple(sorted([remaining[i] for i in idx])))
                    candidates = list(candidates)

                for combo in candidates:
                    sc = score_combination_v(tuple(combo), struct_pred, cfg, pair_z, prev_nums, streaks)
                    if sc > best_sc:
                        best_sc = sc
                        best_combo = list(combo)

                if best_combo is not None:
                    partition.append(sorted(best_combo))
                    total_score += best_sc
                    for n in best_combo:
                        remaining.remove(n)
                else:
                    combo = sorted(remaining[:6])
                    partition.append(combo)
                    total_score += score_combination_v(tuple(combo), struct_pred, cfg, pair_z, prev_nums, streaks)
                    remaining = remaining[6:]

        if len(partition) == 5 and total_score > best_total_score:
            best_total_score = total_score
            best_partition = partition

    if best_partition is None:
        np.random.shuffle(pool)
        best_partition = [sorted(pool[i*6:(i+1)*6]) for i in range(5)]

    return best_partition


# ════════════════════════════════════════════════════════════
# 파라미터화된 apply_adjustments / select_pool
# ════════════════════════════════════════════════════════════

def apply_adjustments_v(scores, df, total, cfg):
    adjusted = scores.copy()
    streak_decay = cfg.get("streak_decay", True)
    if streak_decay:
        for num in range(1, 46):
            streak = 0
            for t in range(total - 1, -1, -1):
                row = df.iloc[t]
                if num in [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"], row["n6"]]:
                    streak += 1
                else:
                    break
            if streak >= 4: adjusted[num - 1] *= 0.3
            elif streak >= 3: adjusted[num - 1] *= 0.6

    recent_5 = set()
    for t in range(max(0, total - 5), total):
        row = df.iloc[t]
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            recent_5.add(int(row[col]))
    cold_pool = set(range(1, 46)) - recent_5
    return adjusted, cold_pool


def select_pool_v(adjusted, cold_pool, cfg):
    pool_size = cfg.get("pool_size", 30)
    cold_min = cfg.get("cold_min", 8)

    sorted_indices = np.argsort(adjusted)[::-1]
    pool = set(int(idx + 1) for idx in sorted_indices[:pool_size])

    cold_in = pool & cold_pool
    if len(cold_in) < cold_min:
        cold_ranked = sorted(cold_pool - pool, key=lambda n: adjusted[n - 1], reverse=True)
        hot_ranked = sorted(pool - cold_pool, key=lambda n: adjusted[n - 1])
        needed = cold_min - len(cold_in)
        for i in range(min(needed, len(cold_ranked), len(hot_ranked))):
            pool.discard(hot_ranked[i])
            pool.add(cold_ranked[i])
    return sorted(pool)


# ════════════════════════════════════════════════════════════
# 빠른 백테스트 (경량)
# ════════════════════════════════════════════════════════════

def fast_backtest(cfg, train_start=300, retrain_interval=50, verbose=False):
    """
    설정 기반 빠른 백테스트.
    반환: {pool_avg, best_avg, grade5_count, grade4_count, grade4p_count, elapsed}
    """
    t_start = time.time()
    df = load_data()
    total = len(df)

    binary_matrix = np.zeros((total, 45), dtype=np.float32)
    for i, row in df.iterrows():
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            binary_matrix[i, int(row[col]) - 1] = 1.0

    struct_df = build_structural_df(df)
    regimes = detect_regimes(struct_df)

    # 앙상블 가중치
    w_xgb = cfg.get("w_xgb", 0.30)
    w_tf = cfg.get("w_tf", 0.30)
    w_scorer = cfg.get("w_scorer", 0.20)

    # XGBoost 하이퍼파라미터
    xgb_n_est = cfg.get("xgb_n_est", 100)
    xgb_depth = cfg.get("xgb_depth", 4)
    xgb_lr = cfg.get("xgb_lr", 0.05)
    xgb_lookback = cfg.get("xgb_lookback", 20)

    # Transformer
    tf_epochs = cfg.get("tf_epochs", 30)
    tf_seq = cfg.get("tf_seq", 30)
    tf_d_model = cfg.get("tf_d_model", 64)
    tf_nhead = cfg.get("tf_nhead", 4)
    tf_nlayers = cfg.get("tf_nlayers", 2)

    # 페어 사용 여부
    use_pair = cfg.get("use_pair", False)

    results = []
    xgb_models = None
    tf_model = None
    struct_predictor = StructuralPredictor()
    scorer = NumberScorer()

    for t in range(train_start, total):
        current_regime = max(r for r in regimes if r <= t)

        # ── 재학습 ──
        if (t - train_start) % retrain_interval == 0:
            # XGBoost
            try:
                regime_data = binary_matrix[current_regime:t]
                if len(regime_data) >= 50:
                    X_xgb, y_xgb = [], []
                    for i in range(30, len(regime_data)):
                        X_xgb.append(regime_data[i - xgb_lookback:i].flatten())
                        y_xgb.append(regime_data[i])
                    X_xgb, y_xgb = np.array(X_xgb), np.array(y_xgb)
                    xgb_models = []
                    for num in range(45):
                        m = xgb.XGBClassifier(
                            n_estimators=xgb_n_est, max_depth=xgb_depth,
                            learning_rate=xgb_lr, scale_pos_weight=39/6,
                            verbosity=0, random_state=42
                        )
                        m.fit(X_xgb, y_xgb[:, num])
                        xgb_models.append(m)
            except Exception:
                pass

            # Transformer (custom architecture)
            try:
                if t >= tf_seq + 50:
                    X, y = [], []
                    for tt in range(tf_seq, t):
                        X.append(binary_matrix[tt - tf_seq:tt])
                        y.append(binary_matrix[tt])
                    X = np.array(X, dtype=np.float32)
                    y_arr = np.array(y, dtype=np.float32)
                    dataset = torch.utils.data.TensorDataset(
                        torch.FloatTensor(X), torch.FloatTensor(y_arr))
                    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

                    tf_model = LottoTransformer(
                        d_model=tf_d_model, nhead=tf_nhead,
                        num_layers=tf_nlayers, dim_ff=tf_d_model*2
                    )
                    optimizer = torch.optim.AdamW(tf_model.parameters(), lr=0.001, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tf_epochs)
                    tf_model.train()
                    for ep in range(tf_epochs):
                        for bx, by in loader:
                            optimizer.zero_grad()
                            pred = tf_model(bx)
                            loss = nn.BCELoss()(pred, by)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(tf_model.parameters(), 1.0)
                            optimizer.step()
                        scheduler.step()
                    tf_model.eval()
            except Exception:
                pass

            try:
                struct_predictor.train(struct_df, t)
            except Exception:
                pass

        # ── 예측 ──
        xgb_probs = np.ones(45) / 45
        if xgb_models and t >= xgb_lookback:
            try:
                seq = binary_matrix[t - xgb_lookback:t].flatten().reshape(1, -1)
                xgb_probs = np.array([m.predict_proba(seq)[0, 1] for m in xgb_models])
            except:
                pass

        tf_probs = np.ones(45) / 45
        if tf_model is not None and t >= tf_seq:
            try:
                seq = binary_matrix[t - tf_seq:t]
                X = torch.FloatTensor(seq).unsqueeze(0)
                with torch.no_grad():
                    tf_probs = tf_model(X).squeeze().numpy()
            except:
                pass

        scorer_probs = scorer.score_numbers(df, t, current_regime)

        combined = w_xgb * xgb_probs + w_tf * tf_probs + w_scorer * scorer_probs
        total_w = w_xgb + w_tf + w_scorer
        if total_w > 0:
            combined = combined / combined.sum()

        struct_pred = struct_predictor.predict(struct_df, t) if struct_predictor.models else {}
        adjusted, cold_pool = apply_adjustments_v(combined, df.iloc[:t], t, cfg)
        pool = select_pool_v(adjusted, cold_pool, cfg)

        # 컨텍스트
        pair_z, prev_nums, bt_streaks = None, None, None
        if use_pair and struct_pred:
            pair_z = build_pair_scores(df, t)
        if struct_pred:
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

        if struct_pred:
            games = partition_v(pool, struct_pred, cfg, pair_z, prev_nums, bt_streaks)
        else:
            p = list(pool)
            np.random.shuffle(p)
            games = [sorted(p[i*6:(i+1)*6]) for i in range(5)]

        actual = set()
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            actual.add(int(df.iloc[t][col]))

        pool_hits = len(actual & set(pool))
        game_hits = [len(actual & set(g)) for g in games]
        best_game = max(game_hits)

        results.append({
            "pool_hits": pool_hits,
            "best_game": best_game,
            "game_hits": game_hits,
        })

    elapsed = time.time() - t_start

    pool_all = [r["pool_hits"] for r in results]
    best_all = [r["best_game"] for r in results]
    n = len(results)
    grade5 = sum(1 for r in results if max(r["game_hits"]) >= 3)
    grade4 = sum(1 for r in results if max(r["game_hits"]) >= 4)
    grade5_hit = sum(1 for r in results if max(r["game_hits"]) == 5)

    return {
        "pool_avg": np.mean(pool_all),
        "best_avg": np.mean(best_all),
        "grade5": grade5,          # 3개+ 적중 횟수
        "grade4": grade4,          # 4개+ 적중 횟수
        "grade5_hit": grade5_hit,  # 5개 적중 횟수
        "n": n,
        "elapsed": elapsed,
    }


# ════════════════════════════════════════════════════════════
# 실험 정의
# ════════════════════════════════════════════════════════════

EXPERIMENTS = {
    # ── A. 기본 v1 필터 (검증된 하드필터만) ──
    "A_baseline_v1": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.30, "w_tf": 0.30, "w_scorer": 0.20,
        "pool_size": 30, "cold_min": 8,
    },

    # ── B. v2 필터 (현재 — 하드+소프트 모두) ──
    "B_current_v2": {
        "hard_decade5": True, "hard_mingap5": True, "hard_endsum10": True,
        "hard_carry4": True, "hard_streak2": True,
        "soft_extra": True, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.30, "w_tf": 0.30, "w_scorer": 0.20,
        "pool_size": 30, "cold_min": 8,
    },

    # ── C. 소프트필터만 강화 (하드 추가 없음) ──
    "C_soft_only": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": True, "soft_weight": 0.7,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.30, "w_tf": 0.30, "w_scorer": 0.20,
        "pool_size": 30, "cold_min": 8,
    },

    # ── D. 앙상블 가중치 변경 (XGB 강화) ──
    "D_xgb_heavy": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.45, "w_tf": 0.25, "w_scorer": 0.15,
        "pool_size": 30, "cold_min": 8,
    },

    # ── E. Transformer 강화 ──
    "E_tf_heavy": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.25, "w_tf": 0.45, "w_scorer": 0.15,
        "pool_size": 30, "cold_min": 8,
    },

    # ── F. Scorer 강화 ──
    "F_scorer_heavy": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.20, "w_tf": 0.20, "w_scorer": 0.40,
        "pool_size": 30, "cold_min": 8,
    },

    # ── G. XGBoost 깊은 모델 ──
    "G_xgb_deep": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.35, "w_tf": 0.30, "w_scorer": 0.20,
        "xgb_n_est": 200, "xgb_depth": 6, "xgb_lr": 0.03,
        "pool_size": 30, "cold_min": 8,
    },

    # ── H. Transformer 큰 모델 ──
    "H_tf_big": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.30, "w_tf": 0.35, "w_scorer": 0.20,
        "tf_d_model": 128, "tf_nhead": 8, "tf_nlayers": 3, "tf_epochs": 50,
        "pool_size": 30, "cold_min": 8,
    },

    # ── I. 구조예측 강화 ──
    "I_struct_strong": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": True, "soft_weight": 0.5,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.8,
        "w_xgb": 0.30, "w_tf": 0.30, "w_scorer": 0.20,
        "pool_size": 30, "cold_min": 8,
    },

    # ── J. 풀 35개 ──
    "J_pool35": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.30, "w_tf": 0.30, "w_scorer": 0.20,
        "pool_size": 35, "cold_min": 10,
        "n_attempts": 50, "n_candidates": 200,
    },

    # ── K. Cold 강제 없음 ──
    "K_no_cold": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.30, "w_tf": 0.30, "w_scorer": 0.20,
        "pool_size": 30, "cold_min": 0, "streak_decay": False,
    },

    # ── L. 최적 조합 후보 (v1필터 + 소프트약화 + 구조강화 + XGB깊은) ──
    "L_best_candidate": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": True, "hard_streak2": False,
        "soft_extra": True, "soft_weight": 0.5,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.7,
        "w_xgb": 0.35, "w_tf": 0.30, "w_scorer": 0.20,
        "xgb_n_est": 200, "xgb_depth": 5, "xgb_lr": 0.03,
        "pool_size": 30, "cold_min": 8,
        "n_attempts": 50, "n_candidates": 150,
    },

    # ════════════════════════════════════════════════
    # Round 2: 스크리닝 결과 기반 최적 조합
    # ════════════════════════════════════════════════

    # ── M. 풀35 수정 (6개씩 고정) + TF강화 ──
    "M_pool35_tf": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.25, "w_tf": 0.45, "w_scorer": 0.15,
        "pool_size": 35, "cold_min": 0, "streak_decay": False,
        "n_attempts": 30, "n_candidates": 100,
    },

    # ── N. 풀35 + Cold없음 + v1필터 ──
    "N_pool35_nocold": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.30, "w_tf": 0.30, "w_scorer": 0.20,
        "pool_size": 35, "cold_min": 0, "streak_decay": False,
        "n_attempts": 30, "n_candidates": 100,
    },

    # ── O. 풀35 + 균등앙상블(1.0) ──
    "O_pool35_equal": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.35, "w_tf": 0.35, "w_scorer": 0.30,
        "pool_size": 35, "cold_min": 0, "streak_decay": False,
        "n_attempts": 30, "n_candidates": 100,
    },

    # ── P. 풀35 + TF강화 + 소프트필터 ──
    "P_pool35_tf_soft": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": True, "soft_weight": 0.5,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.6,
        "w_xgb": 0.25, "w_tf": 0.45, "w_scorer": 0.15,
        "pool_size": 35, "cold_min": 0, "streak_decay": False,
        "n_attempts": 30, "n_candidates": 100,
    },

    # ── Q. 풀35 + XGB강화 ──
    "Q_pool35_xgb": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "w_xgb": 0.45, "w_tf": 0.25, "w_scorer": 0.15,
        "xgb_n_est": 200, "xgb_depth": 5, "xgb_lr": 0.03,
        "pool_size": 35, "cold_min": 0, "streak_decay": False,
        "n_attempts": 30, "n_candidates": 100,
    },

    # ── R. 풀35 + 구조예측강화 ──
    "R_pool35_struct": {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": True, "soft_weight": 0.5,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.8,
        "w_xgb": 0.30, "w_tf": 0.35, "w_scorer": 0.20,
        "pool_size": 35, "cold_min": 0, "streak_decay": False,
        "n_attempts": 30, "n_candidates": 100,
    },
}


if __name__ == "__main__":
    import sys

    # 명령줄에서 특정 실험만 실행 가능
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
    else:
        selected = list(EXPERIMENTS.keys())

    print("=" * 80)
    print("  로또 AI 체계적 실험")
    print(f"  실험 수: {len(selected)}개")
    print("=" * 80)

    all_results = {}

    for name in selected:
        if name not in EXPERIMENTS:
            print(f"\n  ⚠ 실험 '{name}' 없음, 건너뜀")
            continue

        cfg = EXPERIMENTS[name]
        print(f"\n  ▶ [{name}] 실행 중...", flush=True)

        result = fast_backtest(cfg, train_start=300, retrain_interval=50)
        all_results[name] = result

        print(f"    풀={result['pool_avg']:.3f}/6  "
              f"최고게임={result['best_avg']:.3f}  "
              f"5등={result['grade5']}회({result['grade5']/result['n']*100:.1f}%)  "
              f"4등+={result['grade4']}회  "
              f"5개={result['grade5_hit']}회  "
              f"({result['elapsed']:.0f}초)")

    # 순위표
    print(f"\n{'=' * 80}")
    print("  실험 결과 순위표 (4등+ 기준)")
    print(f"{'=' * 80}")
    print(f"  {'순위':>3}  {'실험명':<25}  {'풀':>7}  {'최고게임':>7}  {'5등':>5}  {'4등+':>5}  {'5개':>4}")
    print(f"  {'-'*3}  {'-'*25}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*4}")

    ranked = sorted(all_results.items(),
                    key=lambda x: (x[1]["grade4"], x[1]["grade5"], x[1]["best_avg"]),
                    reverse=True)
    for rank, (name, r) in enumerate(ranked, 1):
        print(f"  {rank:>3}  {name:<25}  {r['pool_avg']:>7.3f}  {r['best_avg']:>7.3f}  "
              f"{r['grade5']:>5}  {r['grade4']:>5}  {r['grade5_hit']:>4}")

    # 최적 결과 저장
    best_name = ranked[0][0]
    print(f"\n  ★ 최적 설정: {best_name}")
    print(f"    설정: {json.dumps(EXPERIMENTS[best_name], indent=2, ensure_ascii=False)}")

    # 결과 저장
    with open(os.path.join(os.path.dirname(__file__), "experiment_results.json"), "w", encoding="utf-8") as f:
        save_data = {}
        for name, r in all_results.items():
            save_data[name] = {**r, "config": EXPERIMENTS[name]}
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n  결과 저장: experiment_results.json")
