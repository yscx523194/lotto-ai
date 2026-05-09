"""
로또 6/45 Smart 패턴 분석 시스템 v2
====================================
핵심 아이디어: 개별 번호가 아닌 "구조적 속성"을 예측한 뒤,
구조에 맞는 조합을 생성하여 검색 공간을 극적으로 줄임.

전략:
1. Regime Detection - 추첨기/공 교체 시점 감지
2. Structural Prediction - 합계, 홀짝비, 연번수 등 구조 예측
3. Number Scoring - 구조 + 번호별 확률을 결합한 점수화
4. Calibrated Ensemble - 전략별 가중치를 log-likelihood로 최적화
5. Transformer Attention - 관련 과거 회차 자동 식별
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import xgboost as xgb
from scipy import stats
from scipy.signal import argrelextrema

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "lotto_cache.json")
MODEL_DIR = os.path.join(BASE_DIR, "models_v2")
os.makedirs(MODEL_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════
# 1. 데이터 로딩 및 전처리
# ════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    rows = []
    for val in cache.values():
        nums = sorted(val["당첨번호"])
        rows.append({
            "round": val["회차"],
            "date": val["추첨일"],
            "n1": nums[0], "n2": nums[1], "n3": nums[2],
            "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "bonus": val["보너스번호"],
        })
    df = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    return df


def extract_structural_features(row) -> dict:
    """한 회차의 구조적 속성 추출"""
    nums = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"], row["n6"]]
    nums_sorted = sorted(nums)

    # 합계
    total_sum = sum(nums)

    # 홀짝 비율
    odd_count = sum(1 for n in nums if n % 2 == 1)

    # 고저 비율 (23 기준)
    high_count = sum(1 for n in nums if n >= 23)

    # 연번 수 (consecutive pairs)
    consec_count = sum(1 for i in range(5) if nums_sorted[i + 1] - nums_sorted[i] == 1)

    # 번호 간격 (gaps)
    gaps = [nums_sorted[i + 1] - nums_sorted[i] for i in range(5)]
    max_gap = max(gaps)
    min_gap = min(gaps)
    avg_gap = np.mean(gaps)

    # 10단위 구간 분포 [1-9, 10-19, 20-29, 30-39, 40-45]
    decades = [0] * 5
    for n in nums:
        if n <= 9: decades[0] += 1
        elif n <= 19: decades[1] += 1
        elif n <= 29: decades[2] += 1
        elif n <= 39: decades[3] += 1
        else: decades[4] += 1

    # AC값 (Arithmetic Complexity) - 번호 쌍의 차이 종류 수
    diffs = set()
    for i in range(6):
        for j in range(i + 1, 6):
            diffs.add(abs(nums_sorted[j] - nums_sorted[i]))
    ac_value = len(diffs) - 5  # 이론적으로 0~10

    # 끝수 분포 (0~9)
    endings = [n % 10 for n in nums]
    unique_endings = len(set(endings))

    # 소수 개수
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
    prime_count = sum(1 for n in nums if n in primes)

    # 3의 배수 개수
    mult3_count = sum(1 for n in nums if n % 3 == 0)

    return {
        "sum": total_sum,
        "odd_count": odd_count,
        "high_count": high_count,
        "consec_count": consec_count,
        "max_gap": max_gap,
        "min_gap": min_gap,
        "avg_gap": avg_gap,
        "ac_value": ac_value,
        "unique_endings": unique_endings,
        "prime_count": prime_count,
        "mult3_count": mult3_count,
        "decade_0": decades[0],
        "decade_1": decades[1],
        "decade_2": decades[2],
        "decade_3": decades[3],
        "decade_4": decades[4],
        "span": nums_sorted[5] - nums_sorted[0],
        "center": np.median(nums),
    }


def build_structural_df(df: pd.DataFrame) -> pd.DataFrame:
    """전체 데이터의 구조적 피처 DataFrame"""
    records = []
    for _, row in df.iterrows():
        feat = extract_structural_features(row)
        feat["round"] = row["round"]
        records.append(feat)
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════
# 2. Regime Detection (체제 감지)
# ════════════════════════════════════════════════════════════

def detect_regimes(struct_df: pd.DataFrame, window=50) -> List[int]:
    """
    추첨기/공 교체로 인한 통계적 변화점(changepoint) 감지.
    CUSUM + KS-test 기반.
    """
    changepoints = []
    key_features = ["sum", "odd_count", "high_count", "ac_value", "avg_gap"]

    for feat in key_features:
        values = struct_df[feat].values

        # Rolling KS-test: 이전 window와 이후 window 비교
        for i in range(window, len(values) - window):
            before = values[i - window:i]
            after = values[i:i + window]
            ks_stat, p_value = stats.ks_2samp(before, after)
            if p_value < 0.001:  # 매우 엄격한 기준
                changepoints.append(i)

    # 클러스터링: 근접한 changepoint를 하나로
    if not changepoints:
        return [0]

    changepoints = sorted(set(changepoints))
    clustered = [changepoints[0]]
    for cp in changepoints[1:]:
        if cp - clustered[-1] > 30:
            clustered.append(cp)

    # 빈도 기반 필터링 (여러 피처에서 감지된 것만)
    cp_counter = Counter()
    for cp in changepoints:
        for c in clustered:
            if abs(cp - c) <= 30:
                cp_counter[c] += 1

    significant = [0] + [cp for cp, cnt in cp_counter.items() if cnt >= 2]
    return sorted(significant)


# ════════════════════════════════════════════════════════════
# 3. Structural Predictor (구조 예측)
# ════════════════════════════════════════════════════════════

class StructuralPredictor:
    """
    다음 회차의 구조적 속성(합계, 홀짝비 등)을 예측.
    각 속성별로 분포를 추정하여 "이 구조가 나올 확률"을 계산.
    """

    def __init__(self):
        self.models = {}
        self.feature_names = [
            "sum", "odd_count", "high_count", "consec_count",
            "ac_value", "unique_endings", "prime_count",
            "decade_0", "decade_1", "decade_2", "decade_3", "decade_4"
        ]

    def _build_lag_features(self, series: np.ndarray, idx: int, lags=[1, 2, 3, 5, 10, 20]) -> dict:
        """시계열 lag 피처 생성"""
        feat = {}
        for lag in lags:
            if idx >= lag:
                feat[f"lag_{lag}"] = series[idx - lag]
            else:
                feat[f"lag_{lag}"] = series[0]

        # Rolling statistics
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

        # Momentum
        if idx >= 5:
            feat["momentum_5"] = series[idx - 1] - series[idx - 5]
        else:
            feat["momentum_5"] = 0

        return feat

    def train(self, struct_df: pd.DataFrame, train_end: int):
        """각 구조적 속성별 XGBoost 회귀 모델 학습"""
        for feat_name in self.feature_names:
            series = struct_df[feat_name].values[:train_end]

            X_list, y_list = [], []
            for i in range(50, len(series)):
                lag_feat = self._build_lag_features(series, i)
                X_list.append(lag_feat)
                y_list.append(series[i])

            X = pd.DataFrame(X_list)
            y = np.array(y_list)

            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
            )
            model.fit(X, y)
            self.models[feat_name] = (model, X.columns.tolist())

    def predict(self, struct_df: pd.DataFrame, target_idx: int) -> dict:
        """다음 회차의 구조적 속성 예측 (점 예측 + 분포)"""
        predictions = {}
        for feat_name in self.feature_names:
            if feat_name not in self.models:
                continue

            model, cols = self.models[feat_name]
            series = struct_df[feat_name].values[:target_idx]
            lag_feat = self._build_lag_features(series, len(series))
            X = pd.DataFrame([lag_feat])[cols]

            pred = model.predict(X)[0]

            # 과거 분포 (std 추정)
            recent = series[-50:] if len(series) >= 50 else series
            std = np.std(recent)

            predictions[feat_name] = {"pred": pred, "std": max(std, 0.1)}

        return predictions


# ════════════════════════════════════════════════════════════
# 4. Number Scorer (번호 점수화)
# ════════════════════════════════════════════════════════════

class NumberScorer:
    """
    개별 번호의 출현 확률을 다각도로 점수화.
    구조 예측과 결합하여 최종 점수 산출.
    """

    def __init__(self):
        pass

    def score_numbers(self, df: pd.DataFrame, target_idx: int,
                      regime_start: int = 0) -> np.ndarray:
        """각 번호(1~45)의 점수 계산 (0~1 스케일)"""
        history = df.iloc[regime_start:target_idx]
        n = len(history)
        if n == 0:
            return np.ones(45) / 45

        scores = np.zeros(45)

        # 모든 출현 번호 수집
        all_nums = []
        for _, row in history.iterrows():
            for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
                all_nums.append(row[col])

        freq = Counter(all_nums)

        for num in range(1, 46):
            idx = num - 1
            f = freq.get(num, 0)

            # === 전략 1: 빈도 기반 (hot numbers) ===
            # 최근 W회차에서의 빈도
            hot_score = 0
            for w in [10, 20, 50]:
                recent = history.tail(w)
                recent_count = 0
                for _, row in recent.iterrows():
                    if num in [row["n1"], row["n2"], row["n3"],
                               row["n4"], row["n5"], row["n6"]]:
                        recent_count += 1
                hot_score += recent_count / w
            hot_score /= 3

            # === 전략 2: 갭 기반 (overdue numbers) ===
            last_seen = 0
            for i in range(len(history) - 1, -1, -1):
                row = history.iloc[i]
                if num in [row["n1"], row["n2"], row["n3"],
                           row["n4"], row["n5"], row["n6"]]:
                    last_seen = len(history) - i
                    break

            avg_gap = n / max(f, 1)
            if last_seen > 0:
                gap_ratio = last_seen / avg_gap  # >1이면 overdue
            else:
                gap_ratio = n / avg_gap

            # gap_ratio가 1.5 이상이면 "due" 상태
            overdue_score = min(gap_ratio / 3.0, 1.0)

            # === 전략 3: 이론적 확률과의 편차 ===
            expected = n * 6 / 45
            chi_deviation = (f - expected) / max(np.sqrt(expected), 1)

            # === 전략 4: 최근 트렌드 (5회차 이동평균의 변화) ===
            if n >= 10:
                recent_5 = 0
                prev_5 = 0
                for i in range(max(0, n - 5), n):
                    row = history.iloc[i]
                    if num in [row[c] for c in ["n1", "n2", "n3", "n4", "n5", "n6"]]:
                        recent_5 += 1
                for i in range(max(0, n - 10), max(0, n - 5)):
                    row = history.iloc[i]
                    if num in [row[c] for c in ["n1", "n2", "n3", "n4", "n5", "n6"]]:
                        prev_5 += 1
                trend_score = (recent_5 - prev_5) / 5 + 0.5
            else:
                trend_score = 0.5

            # === 최종 점수: 가중 결합 ===
            scores[idx] = (
                0.30 * hot_score +
                0.30 * overdue_score +
                0.15 * (0.5 + chi_deviation * 0.1) +
                0.25 * trend_score
            )

        # 정규화
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores


# ════════════════════════════════════════════════════════════
# 5. Transformer 모델
# ════════════════════════════════════════════════════════════

class LottoTransformer(nn.Module):
    """
    Self-Attention 기반 시퀀스 모델.
    어떤 과거 회차가 현재와 관련있는지 자동 학습.
    """
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
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, 45),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, 45)
        seq_len = x.size(1)
        h = self.input_proj(x) + self.pos_embed[:, :seq_len, :]
        h = self.transformer(h)
        return self.fc(h[:, -1, :])


def train_transformer(binary_matrix, train_end, seq_len=30, epochs=40, lr=0.001):
    """Transformer 학습"""
    if train_end < seq_len + 50:
        return None

    X, y = [], []
    for t in range(seq_len, train_end):
        X.append(binary_matrix[t - seq_len:t])
        y.append(binary_matrix[t])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X), torch.FloatTensor(y)
    )
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
# 6. Combination Filter (구조 기반 필터링)
# ════════════════════════════════════════════════════════════

def score_combination(combo: tuple, struct_pred: dict) -> float:
    """
    조합이 예측된 구조적 속성에 얼마나 부합하는지 점수화.
    정규분포 가정 하에 log-likelihood 합산.
    + 실제 데이터에서 발견된 하드 필터 적용.
    """
    nums = sorted(combo)
    total_sum = sum(nums)
    odd_count = sum(1 for n in nums if n % 2 == 1)
    high_count = sum(1 for n in nums if n >= 23)
    consec_count = sum(1 for i in range(5) if nums[i + 1] - nums[i] == 1)

    # 3연번 확인
    has_triple = False
    for i in range(4):
        if nums[i + 1] - nums[i] == 1 and nums[i + 2] - nums[i + 1] == 1:
            has_triple = True

    diffs = set()
    for i in range(6):
        for j in range(i + 1, 6):
            diffs.add(abs(nums[j] - nums[i]))
    ac_value = len(diffs) - 5

    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
    prime_count = sum(1 for n in nums if n in primes)

    endings = [n % 10 for n in nums]
    unique_endings = len(set(endings))
    max_same_ending = max(Counter(endings).values())

    decades = [0] * 5
    for n in nums:
        if n <= 9: decades[0] += 1
        elif n <= 19: decades[1] += 1
        elif n <= 29: decades[2] += 1
        elif n <= 39: decades[3] += 1
        else: decades[4] += 1

    # ══════ 하드 필터 (데이터 분석 기반) ══════
    # 이 조건을 위반하면 극히 낮은 점수 부여 (-100)
    HARD_PENALTY = -100.0

    # 합계 범위: 88.7%가 100~190 (필터: 90~200)
    if total_sum < 90 or total_sum > 200:
        return HARD_PENALTY

    # 홀짝비: 홀0~1, 홀5~6은 합계 8.1%뿐
    if odd_count <= 0 or odd_count >= 6:
        return HARD_PENALTY

    # AC값: AC≤4는 2.5%뿐
    if ac_value <= 4:
        return HARD_PENALTY

    # 같은 끝수 4개 이상은 0.4%
    if max_same_ending >= 4:
        return HARD_PENALTY

    # 소수 5개 이상은 0.7%
    if prime_count >= 5:
        return HARD_PENALTY

    # ══════ 소프트 필터 (감점) ══════
    log_prob = 0.0

    # 3연번: 5.4%뿐 → 소폭 감점
    if has_triple:
        log_prob -= 1.5

    # 홀짝 극단(홀1,홀5): 감점
    if odd_count == 1 or odd_count == 5:
        log_prob -= 0.5

    # 연번 3쌍 이상: 1.8%뿐 → 감점
    if consec_count >= 3:
        log_prob -= 1.0

    # ══════ 구조 예측 적합도 ══════
    checks = {
        "sum": total_sum,
        "odd_count": odd_count,
        "high_count": high_count,
        "consec_count": consec_count,
        "ac_value": ac_value,
        "prime_count": prime_count,
        "unique_endings": unique_endings,
        "decade_0": decades[0],
        "decade_1": decades[1],
        "decade_2": decades[2],
        "decade_3": decades[3],
        "decade_4": decades[4],
    }

    for key, actual_val in checks.items():
        if key in struct_pred:
            pred = struct_pred[key]["pred"]
            std = struct_pred[key]["std"]
            z = (actual_val - pred) / std
            log_prob -= 0.5 * z * z

    return log_prob


# ════════════════════════════════════════════════════════════
# 7. Calibrated Ensemble
# ════════════════════════════════════════════════════════════

def calibrate_weights(results_history: list) -> dict:
    """
    각 전략의 과거 성과(log-likelihood)를 기반으로 가중치 최적화.
    Exponential weighting: 최근 성과에 더 큰 가중치.
    """
    if len(results_history) < 20:
        return {"xgb": 0.25, "transformer": 0.25, "scorer": 0.25, "structural": 0.25}

    strategy_scores = defaultdict(list)
    for r in results_history[-100:]:
        for strategy, probs in r.get("strategy_probs", {}).items():
            if probs is not None:
                actual = r["actual_binary"]
                # Log-likelihood
                ll = 0
                for i in range(45):
                    p = np.clip(probs[i], 1e-6, 1 - 1e-6)
                    if actual[i] == 1:
                        ll += np.log(p)
                    else:
                        ll += np.log(1 - p)
                strategy_scores[strategy].append(ll)

    # Exponential weighted average (decay = 0.95)
    weights = {}
    for strategy, scores in strategy_scores.items():
        decay_weights = np.array([0.95 ** (len(scores) - i - 1) for i in range(len(scores))])
        weights[strategy] = np.average(scores, weights=decay_weights)

    # Softmax
    if weights:
        max_w = max(weights.values())
        exp_w = {k: np.exp(v - max_w) for k, v in weights.items()}
        total = sum(exp_w.values())
        weights = {k: v / total for k, v in exp_w.items()}

    return weights


# ════════════════════════════════════════════════════════════
# 8. Smart Selection (최종 번호 선택)
# ════════════════════════════════════════════════════════════

def smart_select(number_probs: np.ndarray, struct_pred: dict,
                 n_candidates=5000, top_n=6,
                 df=None, target_idx=None) -> list:
    """
    확률 가중 샘플링 + 구조 필터링으로 최적 조합 선택.
    데이터 분석에서 발견된 패턴 적용:
    - 3회+ 연속 출현 번호에 감점
    - cold pool(최근 5회 미출현)에서 최소 2개 포함 강제
    - 하드 필터: 합계, 홀짝, AC값 등
    """
    # ── 연속출현 감쇠 적용 ──
    adjusted_probs = number_probs.copy()

    if df is not None and target_idx is not None and target_idx > 0:
        for num in range(1, 46):
            streak = 0
            for t in range(target_idx - 1, -1, -1):
                row = df.iloc[t]
                if num in [row["n1"], row["n2"], row["n3"],
                           row["n4"], row["n5"], row["n6"]]:
                    streak += 1
                else:
                    break

            # 분석 결과: 3연속→11.1%, 4연속→7.1%
            if streak >= 4:
                adjusted_probs[num - 1] *= 0.3
            elif streak >= 3:
                adjusted_probs[num - 1] *= 0.6

        # cold pool: 최근 5회 미출현 번호 파악
        recent_5 = set()
        for t in range(max(0, target_idx - 5), target_idx):
            row = df.iloc[t]
            for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
                recent_5.add(int(row[col]))
        cold_pool = set(range(1, 46)) - recent_5
    else:
        cold_pool = None

    # 확률 정규화
    probs = np.clip(adjusted_probs, 0.01, 1.0)
    probs = probs / probs.sum()

    # 후보 생성: 확률 가중 샘플링
    candidates = set()
    attempts = 0
    max_attempts = n_candidates * 20
    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        try:
            selected = np.random.choice(45, size=6, replace=False, p=probs)
            combo = tuple(sorted(selected + 1))  # 1-indexed

            # cold pool 강제: 최소 2개 포함 (분석: 평균 2.94개가 cold에서 나옴)
            if cold_pool is not None:
                cold_in_combo = len(set(combo) & cold_pool)
                if cold_in_combo < 2:
                    continue

            candidates.add(combo)
        except:
            continue

    # 각 후보 점수화
    scored = []
    for combo in candidates:
        # 번호 확률 점수
        num_score = sum(number_probs[n - 1] for n in combo)

        # 구조 적합도
        struct_score = score_combination(combo, struct_pred)

        # 최종 점수 (곱으로 결합 → 둘 다 높아야 선택)
        final_score = num_score + struct_score * 0.3

        scored.append((combo, final_score, num_score, struct_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ════════════════════════════════════════════════════════════
# 9. Walk-Forward 백테스트
# ════════════════════════════════════════════════════════════

def walk_forward_backtest_v2(df, binary_matrix, struct_df,
                              train_start=300, retrain_interval=50,
                              seq_len=30):
    """Smart Walk-Forward 백테스트"""
    total = len(df)
    results = []
    calibration_history = []

    xgb_model = None
    tf_model = None
    struct_predictor = StructuralPredictor()
    scorer = NumberScorer()

    # Regime detection
    regimes = detect_regimes(struct_df)
    print(f"  감지된 체제 변화점: {[struct_df.iloc[r]['round'] if r < len(struct_df) else '?' for r in regimes]}")

    print(f"\n{'='*70}")
    print(f"  Smart Walk-Forward (회차 {train_start+1} ~ {total})")
    print(f"  재학습 간격: {retrain_interval}")
    print(f"{'='*70}\n")

    ensemble_weights = {"xgb": 0.3, "transformer": 0.3, "scorer": 0.2, "structural": 0.2}

    for t in range(train_start, total):
        # 현재 체제 시작점
        current_regime = 0
        for r in regimes:
            if r <= t:
                current_regime = r

        # 재학습
        if (t - train_start) % retrain_interval == 0:
            print(f"  [{df.iloc[t]['round']}회차] 재학습 (체제: {df.iloc[current_regime]['round']}회~)...", end="\r")

            # XGBoost (번호별 직접 예측)
            try:
                regime_data = binary_matrix[current_regime:t]
                if len(regime_data) >= 50:
                    X_xgb, y_xgb = [], []
                    for i in range(30, len(regime_data)):
                        seq = regime_data[i-20:i].flatten()
                        X_xgb.append(seq)
                        y_xgb.append(regime_data[i])
                    X_xgb = np.array(X_xgb)
                    y_xgb = np.array(y_xgb)

                    xgb_models = []
                    for num in range(45):
                        m = xgb.XGBClassifier(
                            n_estimators=100, max_depth=4, learning_rate=0.05,
                            scale_pos_weight=39/6, verbosity=0, random_state=42
                        )
                        m.fit(X_xgb, y_xgb[:, num])
                        xgb_models.append(m)
                    xgb_model = xgb_models
            except Exception:
                pass

            # Transformer
            try:
                tf_model = train_transformer(binary_matrix, t, seq_len=seq_len, epochs=30)
            except Exception:
                pass

            # Structural Predictor
            try:
                struct_predictor.train(struct_df, t)
            except Exception:
                pass

            # Calibrate weights
            if len(calibration_history) >= 20:
                ensemble_weights = calibrate_weights(calibration_history)

        # ──── 예측 ────
        strategy_probs = {}

        # XGBoost
        if xgb_model and t >= 20:
            try:
                seq = binary_matrix[t-20:t].flatten().reshape(1, -1)
                xgb_probs = np.array([m.predict_proba(seq)[0, 1] for m in xgb_model])
                strategy_probs["xgb"] = xgb_probs
            except:
                strategy_probs["xgb"] = None
        else:
            strategy_probs["xgb"] = None

        # Transformer
        tf_probs = predict_transformer(tf_model, binary_matrix, t, seq_len)
        strategy_probs["transformer"] = tf_probs

        # Number Scorer
        scorer_probs = scorer.score_numbers(df, t, current_regime)
        strategy_probs["scorer"] = scorer_probs

        # Structural (구조 기반 번호 점수)
        struct_pred = struct_predictor.predict(struct_df, t) if struct_predictor.models else {}

        # ──── 앙상블 ────
        combined = np.zeros(45)
        total_weight = 0
        for strategy, probs in strategy_probs.items():
            if probs is not None:
                w = ensemble_weights.get(strategy, 0.25)
                combined += w * probs
                total_weight += w

        if total_weight > 0:
            combined /= total_weight
        else:
            combined = np.ones(45) / 45

        # Smart selection
        if struct_pred:
            best_combos = smart_select(combined, struct_pred, n_candidates=2000, top_n=1,
                                       df=df, target_idx=t)
            predicted = list(best_combos[0][0]) if best_combos else sorted((np.argsort(combined)[-6:] + 1).tolist())
        else:
            predicted = sorted((np.argsort(combined)[-6:] + 1).tolist())

        actual_set = set()
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            actual_set.add(int(df.iloc[t][col]))

        hits = len(set(predicted) & actual_set)

        result = {
            "round": int(df.iloc[t]["round"]),
            "predicted": predicted,
            "actual": sorted(actual_set),
            "hits": hits,
            "actual_binary": binary_matrix[t],
            "strategy_probs": strategy_probs,
            "weights": dict(ensemble_weights),
        }
        results.append(result)
        calibration_history.append(result)

        if (t - train_start) % 100 == 0 and results:
            recent = results[-min(100, len(results)):]
            avg_hits = np.mean([r["hits"] for r in recent])
            w_str = " ".join(f"{k}:{v:.2f}" for k, v in ensemble_weights.items())
            print(f"  [{df.iloc[t]['round']:>4}회차] 평균 적중: {avg_hits:.2f}/6 | 가중치: {w_str}")

    return results


# ════════════════════════════════════════════════════════════
# 10. 결과 분석
# ════════════════════════════════════════════════════════════

def analyze_results_v2(results):
    if not results:
        print("결과 없음")
        return

    hits = [r["hits"] for r in results]
    total = len(results)
    random_expected = 6 * 6 / 45  # 0.8

    print(f"\n{'='*70}")
    print(f"  백테스트 결과 (총 {total}회)")
    print(f"{'='*70}")

    avg_hits = np.mean(hits)
    print(f"\n  평균 적중: {avg_hits:.3f} / 6  (랜덤 기대값: {random_expected:.3f})")
    print(f"  vs 랜덤:   {'+' if avg_hits > random_expected else ''}{avg_hits - random_expected:.3f}")

    # 분포
    print(f"\n  적중 분포:")
    for h in range(7):
        count = hits.count(h)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        label = ""
        if h == 3: label = " (5등)"
        if h == 4: label = " (4등)"
        if h == 5: label = " (3등)"
        if h == 6: label = " (1등!)"
        print(f"    {h}개: {count:4d}회 ({pct:5.1f}%) {bar}{label}")

    for threshold in [1, 2, 3, 4]:
        pct = sum(1 for h in hits if h >= threshold) / total * 100
        print(f"  {threshold}개+ 적중률: {pct:.1f}%")

    # 시계열 추이 (100회 윈도우)
    if len(results) > 200:
        print(f"\n  시계열 추이:")
        for start in range(0, len(results), 200):
            end = min(start + 200, len(results))
            chunk = results[start:end]
            chunk_avg = np.mean([r["hits"] for r in chunk])
            r_start = chunk[0]["round"]
            r_end = chunk[-1]["round"]
            bar = "█" * int(chunk_avg * 10)
            print(f"    {r_start:>4}~{r_end:>4}회: {chunk_avg:.3f} {bar}")

    # 최종 가중치
    if results[-1].get("weights"):
        print(f"\n  최종 앙상블 가중치:")
        for k, v in results[-1]["weights"].items():
            bar = "█" * int(v * 40)
            print(f"    {k:>12}: {v:.3f} {bar}")

    return {"avg_hits": avg_hits, "total": total,
            "distribution": {h: hits.count(h) for h in range(7)}}


# ════════════════════════════════════════════════════════════
# 11. 다음 회차 예측
# ════════════════════════════════════════════════════════════

def predict_next_v2(df, binary_matrix, struct_df, seq_len=30):
    total = len(df)
    last_round = int(df.iloc[-1]["round"])

    print(f"\n{'='*70}")
    print(f"  {last_round + 1}회차 예측")
    print(f"{'='*70}")

    # Regime
    regimes = detect_regimes(struct_df)
    current_regime = max(r for r in regimes if r < total)
    regime_round = int(struct_df.iloc[current_regime]["round"])
    print(f"\n  현재 체제: {regime_round}회차~")

    # 모델 학습
    print("  모델 학습 중...")

    # XGBoost
    regime_data = binary_matrix[current_regime:total]
    xgb_models = []
    X_xgb = []
    y_xgb = []
    for i in range(30, len(regime_data)):
        X_xgb.append(regime_data[i - 20:i].flatten())
        y_xgb.append(regime_data[i])
    X_xgb = np.array(X_xgb)
    y_xgb = np.array(y_xgb)

    for num in range(45):
        m = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            scale_pos_weight=39/6, verbosity=0, random_state=42
        )
        m.fit(X_xgb, y_xgb[:, num])
        xgb_models.append(m)

    seq = binary_matrix[total - 20:total].flatten().reshape(1, -1)
    xgb_probs = np.array([m.predict_proba(seq)[0, 1] for m in xgb_models])

    # Transformer
    tf_model = train_transformer(binary_matrix, total, seq_len=seq_len, epochs=50)
    tf_probs = predict_transformer(tf_model, binary_matrix, total, seq_len)

    # Number Scorer
    scorer = NumberScorer()
    scorer_probs = scorer.score_numbers(df, total, current_regime)

    # Structural
    struct_predictor = StructuralPredictor()
    struct_predictor.train(struct_df, total)
    struct_pred = struct_predictor.predict(struct_df, total)

    # 앙상블
    combined = 0.30 * xgb_probs + 0.30 * tf_probs + 0.20 * scorer_probs
    # structural은 필터에서 사용
    combined = combined / combined.sum()

    # Smart selection
    top_combos = smart_select(combined, struct_pred, n_candidates=10000, top_n=5,
                              df=df, target_idx=total)

    print(f"\n  {'순위':>4}  {'번호 조합':<30}  {'총점':>8}  {'번호점수':>8}  {'구조점수':>8}")
    print(f"  {'─'*70}")
    for i, (combo, final, num_s, struct_s) in enumerate(top_combos):
        combo_str = str([int(n) for n in combo])
        print(f"  {i+1:>4}  {combo_str:<30}  {final:>8.3f}  {num_s:>8.3f}  {struct_s:>8.3f}")

    best = list(top_combos[0][0])

    # 구조 예측 출력
    print(f"\n  구조 예측:")
    for key, val in struct_pred.items():
        print(f"    {key:>16}: {val['pred']:>6.1f} ± {val['std']:>4.1f}")

    # 선택된 조합의 구조
    print(f"\n  선택 조합의 실제 구조:")
    actual_struct = {
        "sum": sum(best),
        "odd_count": sum(1 for n in best if n % 2 == 1),
        "high_count": sum(1 for n in best if n >= 23),
    }
    for key, val in actual_struct.items():
        pred_val = struct_pred.get(key, {}).get("pred", "?")
        print(f"    {key:>16}: {val} (예측: {pred_val:.1f})" if isinstance(pred_val, float) else f"    {key:>16}: {val}")

    # 저장
    prediction = {
        "target_round": last_round + 1,
        "trained_on": last_round,
        "timestamp": datetime.now().isoformat(),
        "top5_combos": [[int(n) for n in c[0]] for c in top_combos],
        "best_combo": [int(n) for n in best],
        "probabilities": {int(i + 1): float(combined[i]) for i in range(45)},
    }
    with open(os.path.join(MODEL_DIR, "prediction_v2.json"), "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)

    print(f"\n  ★ {last_round + 1}회차 추천: {[int(n) for n in best]}")
    return prediction


# ════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backtest", "predict", "full"], default="full")
    parser.add_argument("--train-start", type=int, default=300)
    parser.add_argument("--retrain", type=int, default=50)
    args = parser.parse_args()

    print("🎱 로또 6/45 Smart 패턴 분석 v2")
    print("=" * 70)

    df = load_data()
    binary_matrix = np.zeros((len(df), 45), dtype=np.float32)
    for i, row in df.iterrows():
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            binary_matrix[i, row[col] - 1] = 1.0

    struct_df = build_structural_df(df)
    print(f"데이터: {len(df)}회차 | 구조 피처: {len(struct_df.columns)}개")

    if args.mode in ["backtest", "full"]:
        results = walk_forward_backtest_v2(
            df, binary_matrix, struct_df,
            train_start=args.train_start,
            retrain_interval=args.retrain,
        )
        bt_stats = analyze_results_v2(results)

    if args.mode in ["predict", "full"]:
        predict_next_v2(df, binary_matrix, struct_df)
