"""
로또 6/45 패턴 분석 및 예측 시스템
- XGBoost 앙상블 (Quant)
- LSTM 시퀀스 모델 (Deep Learning)
- Walk-forward 백테스팅
- 증분 학습 (매주 새 회차 추가)
"""

import json
import os
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

# ============================================================
# 1. 데이터 로딩
# ============================================================

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lotto_cache.json")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def load_data():
    """캐시에서 로또 데이터를 로드하여 DataFrame으로 반환"""
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    rows = []
    for key, val in cache.items():
        rows.append({
            "round": val["회차"],
            "date": val["추첨일"],
            "n1": val["당첨번호"][0],
            "n2": val["당첨번호"][1],
            "n3": val["당첨번호"][2],
            "n4": val["당첨번호"][3],
            "n5": val["당첨번호"][4],
            "n6": val["당첨번호"][5],
            "bonus": val["보너스번호"],
        })

    df = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    return df


def to_binary_matrix(df):
    """각 회차를 45차원 바이너리 벡터로 변환"""
    matrix = np.zeros((len(df), 45), dtype=np.float32)
    for i, row in df.iterrows():
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            matrix[i, row[col] - 1] = 1.0
    return matrix


# ============================================================
# 2. 피처 엔지니어링
# ============================================================

def build_features_for_number(history_matrix, num_idx, windows=[5, 10, 20, 50, 100]):
    """
    특정 번호(0~44)에 대해 피처 벡터 생성.
    history_matrix: 이전 회차들의 45차원 바이너리 매트릭스
    """
    n = len(history_matrix)
    col = history_matrix[:, num_idx]  # 해당 번호의 출현 기록

    features = {}

    # 빈도: 최근 N회차에서 출현 빈도
    for w in windows:
        if n >= w:
            features[f"freq_{w}"] = col[-w:].sum() / w
        else:
            features[f"freq_{w}"] = col.sum() / n if n > 0 else 0

    # 갭: 마지막 출현 이후 경과 회차 수
    appeared = np.where(col == 1)[0]
    if len(appeared) > 0:
        features["gap"] = n - appeared[-1] - 1
        features["avg_gap"] = n / len(appeared)
        features["max_gap"] = max(np.diff(np.concatenate([[-1], appeared]))) - 1 if len(appeared) > 1 else features["gap"]
        features["gap_ratio"] = features["gap"] / features["avg_gap"] if features["avg_gap"] > 0 else 0
    else:
        features["gap"] = n
        features["avg_gap"] = n
        features["max_gap"] = n
        features["gap_ratio"] = 1.0

    # 최근 출현 여부
    for lag in [1, 2, 3, 5]:
        if n >= lag:
            features[f"appeared_{lag}ago"] = float(col[-lag])
        else:
            features[f"appeared_{lag}ago"] = 0.0

    # 전체 출현 비율
    features["total_freq"] = col.sum() / n if n > 0 else 0

    # 번호 자체 (구간 정보)
    features["num_value"] = (num_idx + 1) / 45.0
    features["num_group"] = num_idx // 9  # 0~4 (5개 그룹)

    return features


def build_feature_matrix(binary_matrix, target_idx):
    """
    target_idx 회차를 예측하기 위한 피처 매트릭스 생성.
    각 번호(1~45)에 대해 피처를 생성하여 (45, n_features) 반환.
    """
    history = binary_matrix[:target_idx]
    features_list = []
    for num in range(45):
        feat = build_features_for_number(history, num)
        features_list.append(feat)

    feature_df = pd.DataFrame(features_list)
    return feature_df


def build_context_features(binary_matrix, target_idx):
    """
    전체 컨텍스트 피처 (번호별이 아닌 회차 전체 패턴)
    """
    history = binary_matrix[:target_idx]
    n = len(history)
    if n == 0:
        return {}

    last_draw = history[-1]
    features = {}

    # 직전 회차의 합계
    features["last_sum"] = np.where(last_draw == 1)[0].sum() + 6  # 번호 합 (1-indexed)

    # 직전 회차 홀짝 비율
    last_nums = np.where(last_draw == 1)[0] + 1
    features["last_odd_ratio"] = sum(1 for n in last_nums if n % 2 == 1) / 6

    # 직전 회차 고저 비율
    features["last_high_ratio"] = sum(1 for n in last_nums if n > 22) / 6

    # 최근 10회차 합계 평균
    if n >= 10:
        sums = []
        for i in range(max(0, n - 10), n):
            nums = np.where(history[i] == 1)[0] + 1
            sums.append(nums.sum())
        features["recent_sum_mean"] = np.mean(sums) / 270  # 정규화 (최대합 270)
        features["recent_sum_std"] = np.std(sums) / 270

    return features


# ============================================================
# 3. XGBoost 모델
# ============================================================

def train_xgb_model(binary_matrix, train_end, min_train=100):
    """XGBoost 모델 학습"""
    import xgboost as xgb

    if train_end < min_train:
        return None

    X_list = []
    y_list = []

    for t in range(min_train, train_end):
        feat_df = build_feature_matrix(binary_matrix, t)
        target = binary_matrix[t]  # 45-dim binary

        # 컨텍스트 피처 추가
        ctx = build_context_features(binary_matrix, t)
        for col_name, val in ctx.items():
            feat_df[col_name] = val

        X_list.append(feat_df.values)
        y_list.append(target)

    X = np.vstack(X_list)  # (train_size * 45, n_features)
    y = np.concatenate(y_list)  # (train_size * 45,)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=39 / 6,  # 클래스 불균형 보정
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def predict_xgb(model, binary_matrix, target_idx):
    """XGBoost로 다음 회차 번호 확률 예측"""
    feat_df = build_feature_matrix(binary_matrix, target_idx)
    ctx = build_context_features(binary_matrix, target_idx)
    for col_name, val in ctx.items():
        feat_df[col_name] = val

    probs = model.predict_proba(feat_df.values)[:, 1]
    return probs  # 45-dim probability


# ============================================================
# 4. LSTM 모델
# ============================================================

import torch
import torch.nn as nn


class LottoLSTM(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 45)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, 45)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]  # 마지막 타임스텝
        out = self.relu(self.fc1(self.dropout(last)))
        out = torch.sigmoid(self.fc2(out))
        return out


def prepare_lstm_data(binary_matrix, start, end, seq_len=20):
    """LSTM 학습 데이터 준비"""
    X, y = [], []
    for t in range(start + seq_len, end):
        X.append(binary_matrix[t - seq_len:t])
        y.append(binary_matrix[t])
    return np.array(X), np.array(y)


def train_lstm_model(binary_matrix, train_end, seq_len=20, epochs=50, lr=0.001):
    """LSTM 모델 학습"""
    min_data = seq_len + 50
    if train_end < min_data:
        return None

    X, y = prepare_lstm_data(binary_matrix, 0, train_end, seq_len)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = LottoLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pos_weight = torch.ones(45) * (39 / 6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            # forward에서 sigmoid 제거하고 여기서 logits 사용
            output = model(batch_X)
            # sigmoid가 이미 적용되어 있으므로 BCELoss 사용
            loss = nn.BCELoss()(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    return model


def predict_lstm(model, binary_matrix, target_idx, seq_len=20):
    """LSTM으로 다음 회차 번호 확률 예측"""
    if target_idx < seq_len:
        return np.ones(45) / 45

    seq = binary_matrix[target_idx - seq_len:target_idx]
    X = torch.FloatTensor(seq).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        probs = model(X).squeeze().numpy()
    return probs


# ============================================================
# 5. 앙상블
# ============================================================

def ensemble_predict(xgb_probs, lstm_probs, xgb_weight=0.5):
    """두 모델의 예측을 앙상블"""
    if xgb_probs is None:
        return lstm_probs
    if lstm_probs is None:
        return xgb_probs
    return xgb_weight * xgb_probs + (1 - xgb_weight) * lstm_probs


def select_numbers(probs, n=6):
    """확률 기반으로 상위 N개 번호 선택 (1-indexed)"""
    top_indices = np.argsort(probs)[-n:]
    return sorted(top_indices + 1)  # 1-indexed


# ============================================================
# 6. Walk-Forward 백테스팅
# ============================================================

def walk_forward_backtest(binary_matrix, df,
                          train_start=200,
                          retrain_interval=50,
                          xgb_weight=0.5,
                          seq_len=20,
                          verbose=True):
    """
    Walk-forward 방식 백테스트.
    - train_start부터 시작하여 한 회차씩 전진
    - retrain_interval마다 모델 재학습
    """
    total = len(binary_matrix)
    results = []

    xgb_model = None
    lstm_model = None

    print(f"\n{'='*70}")
    print(f"  Walk-Forward 백테스트 (회차 {train_start+1} ~ {total})")
    print(f"  재학습 간격: {retrain_interval}회차")
    print(f"{'='*70}\n")

    for t in range(train_start, total):
        # 재학습 시점
        if (t - train_start) % retrain_interval == 0:
            if verbose:
                print(f"  [{t+1}회차] 모델 재학습 중 (학습 데이터: 1~{t}회차)...", end="\r")

            try:
                xgb_model = train_xgb_model(binary_matrix, t, min_train=100)
            except Exception as e:
                if verbose:
                    print(f"  XGBoost 학습 실패: {e}")

            try:
                lstm_model = train_lstm_model(binary_matrix, t, seq_len=seq_len, epochs=30)
            except Exception as e:
                if verbose:
                    print(f"  LSTM 학습 실패: {e}")

        # 예측
        xgb_probs = predict_xgb(xgb_model, binary_matrix, t) if xgb_model else None
        lstm_probs = predict_lstm(lstm_model, binary_matrix, t, seq_len) if lstm_model else None

        combined = ensemble_predict(xgb_probs, lstm_probs, xgb_weight)

        if combined is None:
            continue

        predicted = select_numbers(combined, n=6)
        actual_set = set(np.where(binary_matrix[t] == 1)[0] + 1)  # 1-indexed
        predicted_set = set(predicted)
        hits = len(predicted_set & actual_set)

        results.append({
            "round": df.iloc[t]["round"],
            "date": df.iloc[t]["date"],
            "predicted": sorted(predicted),
            "actual": sorted(actual_set),
            "hits": hits,
            "xgb_probs": xgb_probs,
            "lstm_probs": lstm_probs,
        })

        if verbose and (t - train_start) % 100 == 0 and len(results) > 0:
            recent = results[-min(100, len(results)):]
            avg_hits = np.mean([r["hits"] for r in recent])
            print(f"  [{t+1}회차] 최근 평균 적중: {avg_hits:.2f}/6 | 누적 {len(results)}회 테스트 완료")

    return results


def analyze_results(results):
    """백테스트 결과 분석"""
    if not results:
        print("결과 없음")
        return

    hits = [r["hits"] for r in results]
    total = len(results)

    print(f"\n{'='*70}")
    print(f"  백테스트 결과 분석 ({total}회차)")
    print(f"{'='*70}")

    # 기본 통계
    avg_hits = np.mean(hits)
    random_expected = 6 * 6 / 45  # ≈ 0.8

    print(f"\n  평균 적중 수: {avg_hits:.3f} / 6")
    print(f"  랜덤 기대값: {random_expected:.3f} / 6")
    print(f"  vs 랜덤:     {'+' if avg_hits > random_expected else ''}{avg_hits - random_expected:.3f}")

    # 적중 분포
    print(f"\n  적중 분포:")
    for h in range(7):
        count = hits.count(h)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {h}개 적중: {count:4d}회 ({pct:5.1f}%) {bar}")

    # 1개 이상 적중률
    hit_any = sum(1 for h in hits if h >= 1) / total * 100
    hit_2plus = sum(1 for h in hits if h >= 2) / total * 100
    hit_3plus = sum(1 for h in hits if h >= 3) / total * 100

    print(f"\n  1개 이상 적중: {hit_any:.1f}%")
    print(f"  2개 이상 적중: {hit_2plus:.1f}%")
    print(f"  3개 이상 적중: {hit_3plus:.1f}%")

    # 최근 성과 추이 (마지막 100회)
    if len(results) > 100:
        recent = results[-100:]
        recent_avg = np.mean([r["hits"] for r in recent])
        early = results[:100]
        early_avg = np.mean([r["hits"] for r in early])
        print(f"\n  초기 100회 평균: {early_avg:.3f}")
        print(f"  최근 100회 평균: {recent_avg:.3f}")
        improvement = recent_avg - early_avg
        print(f"  학습 개선:       {'+' if improvement > 0 else ''}{improvement:.3f}")

    # 최고 적중 회차
    best = max(results, key=lambda r: r["hits"])
    print(f"\n  최고 적중: {best['round']}회차 - {best['hits']}개 적중")
    print(f"    예측: {best['predicted']}")
    print(f"    실제: {sorted(best['actual'])}")

    return {
        "avg_hits": avg_hits,
        "random_expected": random_expected,
        "total_tests": total,
        "hit_distribution": {h: hits.count(h) for h in range(7)},
    }


# ============================================================
# 7. 다음 회차 예측
# ============================================================

def predict_next_draw(binary_matrix, df, xgb_weight=0.5, seq_len=20, top_n=10):
    """전체 데이터로 학습 후 다음 회차 예측"""
    total = len(binary_matrix)
    last_round = int(df.iloc[-1]["round"])

    print(f"\n{'='*70}")
    print(f"  {last_round + 1}회차 예측 (전체 {total}회차 학습)")
    print(f"{'='*70}")

    # 전체 데이터로 모델 학습
    print("\n  XGBoost 학습 중...")
    xgb_model = train_xgb_model(binary_matrix, total, min_train=100)

    print("  LSTM 학습 중...")
    lstm_model = train_lstm_model(binary_matrix, total, seq_len=seq_len, epochs=50)

    # 예측
    xgb_probs = predict_xgb(xgb_model, binary_matrix, total) if xgb_model else None
    lstm_probs = predict_lstm(lstm_model, binary_matrix, total, seq_len) if lstm_model else None
    combined = ensemble_predict(xgb_probs, lstm_probs, xgb_weight)

    # 상위 번호 출력
    ranking = np.argsort(combined)[::-1] + 1  # 1-indexed, 확률 높은 순
    top6 = sorted(ranking[:6])
    top10 = sorted(ranking[:top_n])

    print(f"\n  추천 번호 (상위 6개): {top6}")
    print(f"  확장 후보 (상위 {top_n}개): {top10}")

    print(f"\n  번호별 확률 (상위 15):")
    print(f"  {'번호':>4}  {'XGBoost':>8}  {'LSTM':>8}  {'앙상블':>8}")
    print(f"  {'─'*36}")
    for i in range(15):
        num = ranking[i]
        xgb_p = xgb_probs[num - 1] if xgb_probs is not None else 0
        lstm_p = lstm_probs[num - 1] if lstm_probs is not None else 0
        comb_p = combined[num - 1]
        marker = " ◀" if num in top6 else ""
        print(f"  {num:4d}  {xgb_p:8.4f}  {lstm_p:8.4f}  {comb_p:8.4f}{marker}")

    # 모델 저장
    os.makedirs(MODEL_DIR, exist_ok=True)
    if xgb_model:
        xgb_model.save_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    if lstm_model:
        torch.save(lstm_model.state_dict(), os.path.join(MODEL_DIR, "lstm_model.pt"))

    # 예측 결과 저장
    prediction = {
        "target_round": last_round + 1,
        "trained_on": last_round,
        "timestamp": datetime.now().isoformat(),
        "top6": [int(x) for x in top6],
        "top10": [int(x) for x in top10],
        "probabilities": {int(i + 1): float(combined[i]) for i in range(45)},
    }
    with open(os.path.join(MODEL_DIR, "latest_prediction.json"), "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)

    print(f"\n  모델 및 예측 저장 완료 → {MODEL_DIR}/")
    return prediction


# ============================================================
# 8. 증분 학습 (새 회차 추가 시)
# ============================================================

def incremental_update(xgb_weight=0.5, seq_len=20):
    """새 회차가 추가되면 증분 학습"""
    pred_path = os.path.join(MODEL_DIR, "latest_prediction.json")

    df = load_data()
    binary_matrix = to_binary_matrix(df)
    last_round = int(df.iloc[-1]["round"])

    if os.path.exists(pred_path):
        with open(pred_path, "r") as f:
            prev = json.load(f)
        prev_round = prev.get("trained_on", 0)

        if last_round > prev_round:
            print(f"새 회차 감지! ({prev_round} → {last_round})")

            # 이전 예측 평가
            if prev.get("target_round") <= last_round:
                target_row = df[df["round"] == prev["target_round"]]
                if len(target_row) > 0:
                    actual = set()
                    row = target_row.iloc[0]
                    for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
                        actual.add(int(row[col]))
                    predicted = set(prev["top6"])
                    hits = len(predicted & actual)
                    print(f"  이전 예측 ({prev['target_round']}회차): {sorted(predicted)}")
                    print(f"  실제 당첨: {sorted(actual)}")
                    print(f"  적중: {hits}/6개")
        else:
            print(f"새 데이터 없음 (현재 {last_round}회차)")
            return

    # 새로 학습 및 예측
    predict_next_draw(binary_matrix, df, xgb_weight, seq_len)


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="로또 6/45 패턴 분석 시스템")
    parser.add_argument("--mode", choices=["backtest", "predict", "update", "full"],
                        default="full", help="실행 모드")
    parser.add_argument("--train-start", type=int, default=200,
                        help="백테스트 시작 회차 (기본: 200)")
    parser.add_argument("--retrain", type=int, default=50,
                        help="재학습 간격 (기본: 50)")
    parser.add_argument("--xgb-weight", type=float, default=0.5,
                        help="XGBoost 가중치 (기본: 0.5)")
    args = parser.parse_args()

    print("🎱 로또 6/45 패턴 분석 시스템")
    print("=" * 70)

    # 데이터 로드
    df = load_data()
    binary_matrix = to_binary_matrix(df)
    print(f"데이터 로드: {len(df)}회차 (1회 ~ {int(df.iloc[-1]['round'])}회)")

    if args.mode in ["backtest", "full"]:
        results = walk_forward_backtest(
            binary_matrix, df,
            train_start=args.train_start,
            retrain_interval=args.retrain,
            xgb_weight=args.xgb_weight,
        )
        stats = analyze_results(results)

        # 결과 저장
        os.makedirs(MODEL_DIR, exist_ok=True)
        backtest_summary = {
            "timestamp": datetime.now().isoformat(),
            "train_start": args.train_start,
            "retrain_interval": args.retrain,
            "xgb_weight": args.xgb_weight,
            "stats": stats,
        }
        with open(os.path.join(MODEL_DIR, "backtest_results.json"), "w", encoding="utf-8") as f:
            json.dump(backtest_summary, f, ensure_ascii=False, indent=2, default=str)

    if args.mode in ["predict", "full"]:
        predict_next_draw(binary_matrix, df, args.xgb_weight)

    if args.mode == "update":
        incremental_update(args.xgb_weight)
