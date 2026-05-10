"""
로또 6/45 모델 학습 (로컬 전용)
================================
학습 결과를 lotto_models/meta.json에 저장.
운영에서는 predict_fast.py가 이 파일만 읽어서 예측.

사용법:
    python train_model.py          # 학습 + 저장
    python train_model.py --test   # 학습 + 테스트 예측까지

의존성: numpy, pandas, torch, xgboost, scipy
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import xgboost as xgb
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "lotto_cache.json")
MODEL_DIR = os.path.join(BASE_DIR, "lotto_models")


# ════════════════════════════════════════════════════════════
# 데이터 로딩
# ════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    rows = []
    for val in cache.values():
        nums = sorted(val["당첨번호"])
        rows.append({
            "round": val["회차"],
            "n1": nums[0], "n2": nums[1], "n3": nums[2],
            "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "bonus": val["보너스번호"],
        })
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def extract_structural_features(row) -> dict:
    nums = sorted([row["n1"], row["n2"], row["n3"], row["n4"], row["n5"], row["n6"]])
    total_sum = sum(nums)
    odd_count = sum(1 for n in nums if n % 2 == 1)
    high_count = sum(1 for n in nums if n >= 23)
    consec_count = sum(1 for i in range(5) if nums[i + 1] - nums[i] == 1)
    gaps = [nums[i + 1] - nums[i] for i in range(5)]

    decades = [0] * 5
    for n in nums:
        if n <= 9: decades[0] += 1
        elif n <= 19: decades[1] += 1
        elif n <= 29: decades[2] += 1
        elif n <= 39: decades[3] += 1
        else: decades[4] += 1

    diffs = set()
    for i in range(6):
        for j in range(i + 1, 6):
            diffs.add(abs(nums[j] - nums[i]))
    ac_value = len(diffs) - 5

    endings = [n % 10 for n in nums]
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
    prime_count = sum(1 for n in nums if n in primes)

    return {
        "sum": total_sum, "odd_count": odd_count, "high_count": high_count,
        "consec_count": consec_count, "ac_value": ac_value,
        "unique_endings": len(set(endings)), "prime_count": prime_count,
        "decade_0": decades[0], "decade_1": decades[1], "decade_2": decades[2],
        "decade_3": decades[3], "decade_4": decades[4],
        "max_gap": max(gaps), "min_gap": min(gaps), "avg_gap": np.mean(gaps),
        "span": nums[5] - nums[0], "center": np.median(nums),
    }


def build_structural_df(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        feat = extract_structural_features(row)
        feat["round"] = row["round"]
        records.append(feat)
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════
# Regime Detection
# ════════════════════════════════════════════════════════════

def detect_regimes(struct_df: pd.DataFrame, window=50) -> List[int]:
    changepoints = []
    key_features = ["sum", "odd_count", "high_count", "ac_value", "avg_gap"]
    for feat in key_features:
        values = struct_df[feat].values
        for i in range(window, len(values) - window):
            before = values[i - window:i]
            after = values[i:i + window]
            _, p_value = stats.ks_2samp(before, after)
            if p_value < 0.001:
                changepoints.append(i)

    if not changepoints:
        return [0]

    changepoints = sorted(set(changepoints))
    clustered = [changepoints[0]]
    for cp in changepoints[1:]:
        if cp - clustered[-1] > 30:
            clustered.append(cp)

    cp_counter = Counter()
    for cp in changepoints:
        for c in clustered:
            if abs(cp - c) <= 30:
                cp_counter[c] += 1

    return sorted([0] + [cp for cp, cnt in cp_counter.items() if cnt >= 2])


# ════════════════════════════════════════════════════════════
# Structural Predictor
# ════════════════════════════════════════════════════════════

class StructuralPredictor:
    def __init__(self):
        self.models = {}
        self.feature_names = [
            "sum", "odd_count", "high_count", "consec_count",
            "ac_value", "unique_endings", "prime_count",
            "decade_0", "decade_1", "decade_2", "decade_3", "decade_4"
        ]

    def _build_lag_features(self, series, idx, lags=[1, 2, 3, 5, 10, 20]):
        feat = {}
        for lag in lags:
            feat[f"lag_{lag}"] = series[idx - lag] if idx >= lag else series[0]
        for w in [5, 10, 20, 50]:
            if idx >= w:
                window = series[idx - w:idx]
                feat[f"roll_mean_{w}"] = np.mean(window)
                feat[f"roll_std_{w}"] = np.std(window)
                feat[f"roll_min_{w}"] = np.min(window)
                feat[f"roll_max_{w}"] = np.max(window)
            else:
                feat[f"roll_mean_{w}"] = np.mean(series[:idx]) if idx > 0 else 0
                feat[f"roll_std_{w}"] = np.std(series[:idx]) if idx > 1 else 0
                feat[f"roll_min_{w}"] = np.min(series[:idx]) if idx > 0 else 0
                feat[f"roll_max_{w}"] = np.max(series[:idx]) if idx > 0 else 0
        feat["momentum_5"] = (series[idx - 1] - series[idx - 5]) if idx >= 5 else 0
        return feat

    def train(self, struct_df, train_end):
        for feat_name in self.feature_names:
            series = struct_df[feat_name].values[:train_end]
            X_list, y_list = [], []
            for i in range(50, len(series)):
                X_list.append(self._build_lag_features(series, i))
                y_list.append(series[i])
            X = pd.DataFrame(X_list)
            y = np.array(y_list)
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
            )
            model.fit(X, y)
            self.models[feat_name] = (model, X.columns.tolist())

    def predict(self, struct_df, target_idx):
        predictions = {}
        for feat_name in self.feature_names:
            if feat_name not in self.models:
                continue
            model, cols = self.models[feat_name]
            series = struct_df[feat_name].values[:target_idx]
            lag_feat = self._build_lag_features(series, len(series))
            X = pd.DataFrame([lag_feat])[cols]
            pred = model.predict(X)[0]
            recent = series[-50:] if len(series) >= 50 else series
            std = np.std(recent)
            predictions[feat_name] = {"pred": float(pred), "std": max(float(std), 0.1)}
        return predictions


# ════════════════════════════════════════════════════════════
# Number Scorer
# ════════════════════════════════════════════════════════════

class NumberScorer:
    def score_numbers(self, df, target_idx, regime_start=0):
        history = df.iloc[regime_start:target_idx]
        n = len(history)
        if n == 0:
            return np.ones(45) / 45

        scores = np.zeros(45)
        all_nums = []
        for _, row in history.iterrows():
            for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
                all_nums.append(row[col])
        freq = Counter(all_nums)

        for num in range(1, 46):
            idx = num - 1
            f = freq.get(num, 0)

            # Hot score
            hot_score = 0
            for w in [10, 20, 50]:
                recent = history.tail(w)
                cnt = sum(1 for _, row in recent.iterrows()
                          if num in [row["n1"], row["n2"], row["n3"],
                                     row["n4"], row["n5"], row["n6"]])
                hot_score += cnt / w
            hot_score /= 3

            # Overdue score
            last_seen = 0
            for i in range(len(history) - 1, -1, -1):
                row = history.iloc[i]
                if num in [row["n1"], row["n2"], row["n3"],
                           row["n4"], row["n5"], row["n6"]]:
                    last_seen = len(history) - i
                    break
            avg_gap = n / max(f, 1)
            gap_ratio = (last_seen if last_seen > 0 else n) / avg_gap
            overdue_score = min(gap_ratio / 3.0, 1.0)

            # Chi deviation
            expected = n * 6 / 45
            chi_deviation = (f - expected) / max(np.sqrt(expected), 1)

            # Trend
            if n >= 10:
                recent_5 = sum(1 for i in range(max(0, n - 5), n)
                               if num in [history.iloc[i][c] for c in ["n1", "n2", "n3", "n4", "n5", "n6"]])
                prev_5 = sum(1 for i in range(max(0, n - 10), max(0, n - 5))
                             if num in [history.iloc[i][c] for c in ["n1", "n2", "n3", "n4", "n5", "n6"]])
                trend_score = (recent_5 - prev_5) / 5 + 0.5
            else:
                trend_score = 0.5

            scores[idx] = (0.30 * hot_score + 0.30 * overdue_score +
                           0.15 * (0.5 + chi_deviation * 0.1) + 0.25 * trend_score)

        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores


# ════════════════════════════════════════════════════════════
# Transformer
# ════════════════════════════════════════════════════════════

class LottoTransformer(nn.Module):
    def __init__(self, input_dim=45, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, 45), nn.Sigmoid()
        )

    def forward(self, x):
        seq_len = x.size(1)
        h = self.input_proj(x) + self.pos_embed[:, :seq_len, :]
        h = self.transformer(h)
        return self.fc(h[:, -1, :])


def train_transformer(binary_matrix, train_end, seq_len=30, epochs=40, lr=0.001):
    if train_end < seq_len + 50:
        return None
    X, y = [], []
    for t in range(seq_len, train_end):
        X.append(binary_matrix[t - seq_len:t])
        y.append(binary_matrix[t])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = LottoTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = nn.BCELoss()(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


def predict_transformer(model, binary_matrix, target_idx, seq_len=30):
    if model is None or target_idx < seq_len:
        return None
    seq = binary_matrix[target_idx - seq_len:target_idx]
    X = torch.FloatTensor(seq).unsqueeze(0)
    with torch.no_grad():
        return model(X).squeeze().numpy()


# ════════════════════════════════════════════════════════════
# 조합 점수 + 필터
# ════════════════════════════════════════════════════════════

def score_combination(combo, struct_pred):
    nums = sorted(combo)
    total_sum = sum(nums)
    odd_count = sum(1 for n in nums if n % 2 == 1)
    consec_count = sum(1 for i in range(5) if nums[i + 1] - nums[i] == 1)
    high_count = sum(1 for n in nums if n >= 23)
    has_triple = any(nums[i+1] - nums[i] == 1 and nums[i+2] - nums[i+1] == 1 for i in range(4))

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

    decades = [0] * 5
    for n in nums:
        if n <= 9: decades[0] += 1
        elif n <= 19: decades[1] += 1
        elif n <= 29: decades[2] += 1
        elif n <= 39: decades[3] += 1
        else: decades[4] += 1

    # 하드 필터
    if total_sum < 90 or total_sum > 200: return -100.0
    if odd_count <= 0 or odd_count >= 6: return -100.0
    if ac_value <= 4: return -100.0
    if max_same_ending >= 4: return -100.0
    if prime_count >= 5: return -100.0

    # 소프트 필터
    log_prob = 0.0
    if has_triple: log_prob -= 1.5
    if odd_count == 1 or odd_count == 5: log_prob -= 0.5
    if consec_count >= 3: log_prob -= 1.0

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
            log_prob -= 0.5 * z * z

    return log_prob


# ════════════════════════════════════════════════════════════
# 커버리지 전략 (30개 풀 → 5게임)
# ════════════════════════════════════════════════════════════

def apply_adjustments(scores, df, total):
    adjusted = scores.copy()
    for num in range(1, 46):
        streak = 0
        for t in range(total - 1, -1, -1):
            row = df.iloc[t]
            if num in [row["n1"], row["n2"], row["n3"],
                       row["n4"], row["n5"], row["n6"]]:
                streak += 1
            else:
                break
        if streak >= 4: adjusted[num - 1] *= 0.3
        elif streak >= 3: adjusted[num - 1] *= 0.6

    # cold pool (최근 5회 미출현)
    recent_5 = set()
    for t in range(max(0, total - 5), total):
        row = df.iloc[t]
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            recent_5.add(int(row[col]))
    cold_pool = set(range(1, 46)) - recent_5

    return adjusted, cold_pool


def select_pool_30(adjusted, cold_pool):
    sorted_indices = np.argsort(adjusted)[::-1]
    pool_30 = set(int(idx + 1) for idx in sorted_indices[:30])

    cold_in_pool = pool_30 & cold_pool
    if len(cold_in_pool) < 8:
        cold_ranked = sorted(cold_pool - pool_30,
                             key=lambda n: adjusted[n - 1], reverse=True)
        hot_ranked = sorted(pool_30 - cold_pool,
                            key=lambda n: adjusted[n - 1])
        needed = 8 - len(cold_in_pool)
        for i in range(min(needed, len(cold_ranked), len(hot_ranked))):
            pool_30.discard(hot_ranked[i])
            pool_30.add(cold_ranked[i])

    return sorted(pool_30)


def partition_into_5_games(pool_30, struct_pred, n_attempts=200, n_candidates=500):
    pool = list(pool_30)
    best_partition = None
    best_total_score = -999

    for _ in range(n_attempts):
        partition = []
        remaining = list(pool)
        np.random.shuffle(remaining)
        total_score = 0

        for game_idx in range(5):
            if len(remaining) < 6:
                break
            if game_idx == 4:
                combo = sorted(remaining)
                partition.append(combo)
                total_score += score_combination(tuple(combo), struct_pred)
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
                    sc = score_combination(tuple(combo), struct_pred)
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
                    total_score += score_combination(tuple(combo), struct_pred)
                    remaining = remaining[6:]

        if len(partition) == 5 and total_score > best_total_score:
            best_total_score = total_score
            best_partition = partition

    if best_partition is None:
        np.random.shuffle(pool)
        best_partition = [sorted(pool[i*6:(i+1)*6]) for i in range(5)]

    return best_partition


# ════════════════════════════════════════════════════════════
# 학습 + 저장 메인
# ════════════════════════════════════════════════════════════

def train_and_save():
    """모든 모델 학습 → meta.json에 결과 캐시"""
    print("=" * 60)
    print("  로또 AI 모델 학습")
    print("=" * 60)

    # 데이터 로딩
    df = load_data()
    total = len(df)
    last_round = int(df.iloc[-1]["round"])
    print(f"\n  데이터: {total}회차 (최신: {last_round}회차)")

    # 바이너리 매트릭스
    binary_matrix = np.zeros((total, 45), dtype=np.float32)
    for i, row in df.iterrows():
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            binary_matrix[i, int(row[col]) - 1] = 1.0

    # Regime detection
    struct_df = build_structural_df(df)
    regimes = detect_regimes(struct_df)
    current_regime = max(r for r in regimes if r < total)
    print(f"  현재 체제: {int(df.iloc[current_regime]['round'])}회차~")

    # 1. XGBoost (45개 번호별)
    print("  [1/4] XGBoost 학습...")
    regime_data = binary_matrix[current_regime:total]
    X_xgb, y_xgb = [], []
    for i in range(30, len(regime_data)):
        X_xgb.append(regime_data[i - 20:i].flatten())
        y_xgb.append(regime_data[i])
    X_xgb, y_xgb = np.array(X_xgb), np.array(y_xgb)

    xgb_models = []
    for num in range(45):
        m = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            scale_pos_weight=39/6, verbosity=0, random_state=42
        )
        m.fit(X_xgb, y_xgb[:, num])
        xgb_models.append(m)

    seq = binary_matrix[total - 20:total].flatten().reshape(1, -1)
    xgb_probs = np.array([m.predict_proba(seq)[0, 1] for m in xgb_models])

    # 2. Transformer
    print("  [2/4] Transformer 학습...")
    tf_model = train_transformer(binary_matrix, total, seq_len=30, epochs=50)
    tf_probs = predict_transformer(tf_model, binary_matrix, total, seq_len=30)

    # 3. Number Scorer
    print("  [3/4] Number Scorer...")
    scorer = NumberScorer()
    scorer_probs = scorer.score_numbers(df, total, current_regime)

    # 4. Structural Predictor
    print("  [4/4] Structural Predictor...")
    struct_predictor = StructuralPredictor()
    struct_predictor.train(struct_df, total)
    struct_pred = struct_predictor.predict(struct_df, total)

    # 앙상블
    combined = 0.30 * xgb_probs + 0.30 * tf_probs + 0.20 * scorer_probs
    combined = combined / combined.sum()

    # 연속출현 감쇠 + cold pool
    adjusted, cold_pool = apply_adjustments(combined, df, total)

    # 풀 30개 선정
    pool_30 = select_pool_30(adjusted, cold_pool)
    excluded_15 = sorted(set(range(1, 46)) - set(pool_30))
    cold_in_pool = len(set(pool_30) & cold_pool)

    # 5게임 분할
    games = partition_into_5_games(pool_30, struct_pred)

    # ── meta.json 저장 ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    meta = {
        "trained_on": last_round,
        "target_round": last_round + 1,
        "current_regime": int(df.iloc[current_regime]["round"]),
        "scores": {str(i + 1): round(float(combined[i]), 6) for i in range(45)},
        "adjusted_scores": {str(i + 1): round(float(adjusted[i]), 6) for i in range(45)},
        "struct_pred": {k: {"pred": round(float(v["pred"]), 4),
                            "std": round(float(v["std"]), 4)}
                        for k, v in struct_pred.items()},
        "pool_30": pool_30,
        "excluded_15": excluded_15,
        "cold_in_pool": cold_in_pool,
        "games": games,
    }

    meta_path = os.path.join(MODEL_DIR, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n  저장 완료: {meta_path}")
    print(f"\n{'=' * 60}")
    print(f"  {last_round + 1}회차 예측 (최대 커버리지)")
    print(f"{'=' * 60}")
    print(f"\n  풀 30개: {pool_30}")
    print(f"  제외 15개: {excluded_15}")
    print(f"  cold 포함: {cold_in_pool}개")
    print(f"\n  5게임:")
    for i, g in enumerate(games, 1):
        s = sum(g)
        odd = sum(1 for n in g if n % 2 == 1)
        print(f"    게임{i}: {g}  합={s} 홀={odd}")
    print()

    return meta


if __name__ == "__main__":
    import time
    t0 = time.time()
    meta = train_and_save()
    elapsed = time.time() - t0
    print(f"  학습 소요시간: {elapsed:.1f}초")

    if "--test" in sys.argv:
        print("\n  [테스트] predict_fast.py로 예측...")
        from predict_fast import predict
        result = predict()
        print(f"  predict 결과: {len(result['games'])}게임")
