"""
고급 딥러닝 모델 실험
=====================
LSTM, GRU, Bidirectional, Attention, 깊은 Transformer 등
다양한 아키텍처를 실험한다.
"""
import os, json, time, sys
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import (
    load_data, build_structural_df, detect_regimes,
    StructuralPredictor, NumberScorer,
    build_pair_scores,
)
from experiment import (
    score_combination_v, apply_adjustments_v,
    select_pool_v, partition_v,
)


# ════════════════════════════════════════════════════════════
# LSTM 모델
# ════════════════════════════════════════════════════════════

class LottoLSTM(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True, bidirectional=bidirectional,
        )
        factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * factor, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 45),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ════════════════════════════════════════════════════════════
# GRU 모델
# ════════════════════════════════════════════════════════════

class LottoGRU(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 45),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ════════════════════════════════════════════════════════════
# Self-Attention LSTM (LSTM + Attention)
# ════════════════════════════════════════════════════════════

class LottoLSTMAttention(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 45),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        attn_scores = self.attn_w(lstm_out).squeeze(-1)  # (B, T)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, H)
        return self.fc(context)


# ════════════════════════════════════════════════════════════
# 깊은 Transformer (더 큰 아키텍처)
# ════════════════════════════════════════════════════════════

class LottoDeepTransformer(nn.Module):
    def __init__(self, input_dim=45, d_model=128, nhead=8, num_layers=4,
                 dim_ff=256, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, 45), nn.Sigmoid()
        )

    def forward(self, x):
        seq_len = x.size(1)
        h = self.input_proj(x) + self.pos_embed[:, :seq_len, :]
        h = self.transformer(h)
        h = self.norm(h[:, -1, :])
        return self.fc(h)


# ════════════════════════════════════════════════════════════
# Conv1D + Transformer 하이브리드
# ════════════════════════════════════════════════════════════

class LottoConvTransformer(nn.Module):
    def __init__(self, input_dim=45, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.pos_embed = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, 45), nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, 45) → conv expects (B, C, T)
        h = x.permute(0, 2, 1)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = h.permute(0, 2, 1)  # (B, T, d_model)
        seq_len = h.size(1)
        h = h + self.pos_embed[:, :seq_len, :]
        h = self.transformer(h)
        return self.fc(h[:, -1, :])


# ════════════════════════════════════════════════════════════
# 통합 학습 함수
# ════════════════════════════════════════════════════════════

def train_dl_model(model, binary_matrix, train_end, seq_len=30, epochs=40,
                   lr=0.001, weight_decay=1e-4, label_smoothing=0.0):
    """범용 딥러닝 모델 학습"""
    if train_end < seq_len + 50:
        return None

    X, y = [], []
    for t in range(seq_len, train_end):
        X.append(binary_matrix[t - seq_len:t])
        y.append(binary_matrix[t])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    if label_smoothing > 0:
        y = y * (1 - label_smoothing) + (1 - y) * label_smoothing

    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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


def predict_dl_model(model, binary_matrix, target_idx, seq_len=30):
    """범용 딥러닝 모델 예측"""
    if model is None or target_idx < seq_len:
        return None
    seq = binary_matrix[target_idx - seq_len:target_idx]
    X = torch.FloatTensor(seq).unsqueeze(0)
    with torch.no_grad():
        return model(X).squeeze().numpy()


# ════════════════════════════════════════════════════════════
# DL 모델 백테스트
# ════════════════════════════════════════════════════════════

DL_CONFIGS = {
    "LSTM_128_2L": {
        "model_class": "LottoLSTM",
        "model_args": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.2},
        "train_args": {"epochs": 40, "lr": 0.001, "seq_len": 30},
    },
    "LSTM_256_3L": {
        "model_class": "LottoLSTM",
        "model_args": {"hidden_dim": 256, "num_layers": 3, "dropout": 0.3},
        "train_args": {"epochs": 50, "lr": 0.0008, "seq_len": 30},
    },
    "BiLSTM_128": {
        "model_class": "LottoLSTM",
        "model_args": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.2, "bidirectional": True},
        "train_args": {"epochs": 40, "lr": 0.001, "seq_len": 30},
    },
    "GRU_128_2L": {
        "model_class": "LottoGRU",
        "model_args": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.2},
        "train_args": {"epochs": 40, "lr": 0.001, "seq_len": 30},
    },
    "LSTM_Attn_128": {
        "model_class": "LottoLSTMAttention",
        "model_args": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.2},
        "train_args": {"epochs": 40, "lr": 0.001, "seq_len": 30},
    },
    "DeepTF_128_4L": {
        "model_class": "LottoDeepTransformer",
        "model_args": {"d_model": 128, "nhead": 8, "num_layers": 4, "dim_ff": 256, "dropout": 0.15},
        "train_args": {"epochs": 50, "lr": 0.0008, "seq_len": 30},
    },
    "ConvTF_64_2L": {
        "model_class": "LottoConvTransformer",
        "model_args": {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_ff": 128, "dropout": 0.2},
        "train_args": {"epochs": 40, "lr": 0.001, "seq_len": 30},
    },
    "LSTM_LS": {
        "model_class": "LottoLSTM",
        "model_args": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.2},
        "train_args": {"epochs": 40, "lr": 0.001, "seq_len": 30, "label_smoothing": 0.05},
    },
    "LSTM_seq50": {
        "model_class": "LottoLSTM",
        "model_args": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.2},
        "train_args": {"epochs": 40, "lr": 0.001, "seq_len": 50},
    },
}

MODEL_CLASSES = {
    "LottoLSTM": LottoLSTM,
    "LottoGRU": LottoGRU,
    "LottoLSTMAttention": LottoLSTMAttention,
    "LottoDeepTransformer": LottoDeepTransformer,
    "LottoConvTransformer": LottoConvTransformer,
}


def dl_backtest(dl_name, dl_cfg, train_start=800, retrain_interval=50):
    """
    DL 모델 단독 예측 성능 평가 (앙상블 없이 DL 모델만의 pool hit rate 측정)
    """
    df = load_data()
    total = len(df)

    binary_matrix = np.zeros((total, 45), dtype=np.float32)
    for i, row in df.iterrows():
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            binary_matrix[i, int(row[col]) - 1] = 1.0

    seq_len = dl_cfg["train_args"]["seq_len"]
    epochs = dl_cfg["train_args"]["epochs"]
    lr = dl_cfg["train_args"]["lr"]
    label_smoothing = dl_cfg["train_args"].get("label_smoothing", 0.0)
    model_cls = MODEL_CLASSES[dl_cfg["model_class"]]

    dl_model = None
    results = []
    t_start = time.time()

    for t in range(train_start, total):
        # 재학습
        if (t - train_start) % retrain_interval == 0:
            try:
                dl_model = model_cls(**dl_cfg["model_args"])
                dl_model = train_dl_model(
                    dl_model, binary_matrix, t, seq_len=seq_len,
                    epochs=epochs, lr=lr, label_smoothing=label_smoothing,
                )
            except Exception as e:
                dl_model = None

        # 예측
        probs = predict_dl_model(dl_model, binary_matrix, t, seq_len=seq_len)
        if probs is None:
            probs = np.ones(45) / 45

        # Top 35 선정 (Round 1에서 pool 35가 최적)
        top_idx = np.argsort(probs)[::-1][:35]
        pool = set(int(idx + 1) for idx in top_idx)

        actual = set()
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            actual.add(int(df.iloc[t][col]))

        pool_hits = len(actual & pool)
        results.append(pool_hits)

    elapsed = time.time() - t_start
    avg = np.mean(results)
    return {"pool_avg": avg, "n": len(results), "elapsed": elapsed}


def dl_ensemble_backtest(top_models, train_start=800, retrain_interval=50):
    """
    상위 DL 모델들을 앙상블하여 전체 파이프라인 백테스트
    XGBoost + 상위DL + Scorer → pool → 5게임
    """
    from train_model import LottoTransformer

    df = load_data()
    total = len(df)

    binary_matrix = np.zeros((total, 45), dtype=np.float32)
    for i, row in df.iterrows():
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            binary_matrix[i, int(row[col]) - 1] = 1.0

    struct_df = build_structural_df(df)
    regimes = detect_regimes(struct_df)

    scorer = NumberScorer()
    struct_predictor = StructuralPredictor()
    xgb_models = None

    # DL 모델 컨테이너
    dl_models = {name: None for name in top_models}
    dl_configs_local = {name: DL_CONFIGS[name] for name in top_models}

    cfg = {
        "hard_decade5": False, "hard_mingap5": False, "hard_endsum10": False,
        "hard_carry4": False, "hard_streak2": False,
        "soft_extra": False, "soft_weight": 1.0,
        "use_pair": False, "pair_weight": 0.0, "struct_weight": 0.5,
        "pool_size": 35, "cold_min": 0, "streak_decay": False,
        "n_attempts": 80, "n_candidates": 200,
    }

    results = []
    n_dl = len(top_models)

    # 가중치: XGBoost 0.30 + DL models (합 0.45) + Scorer 0.25 = 1.0
    w_xgb = 0.30
    w_dl_each = 0.45 / max(n_dl, 1)
    w_scorer = 0.25

    for t in range(train_start, total):
        current_regime = max(r for r in regimes if r <= t)

        if (t - train_start) % retrain_interval == 0:
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
                            n_estimators=150, max_depth=5, learning_rate=0.05,
                            scale_pos_weight=39/6, verbosity=0, random_state=42
                        )
                        m.fit(X_xgb, y_xgb[:, num])
                        xgb_models.append(m)
            except:
                pass

            # DL models
            for name, dcfg in dl_configs_local.items():
                try:
                    cls = MODEL_CLASSES[dcfg["model_class"]]
                    model = cls(**dcfg["model_args"])
                    sl = dcfg["train_args"]["seq_len"]
                    ep = dcfg["train_args"]["epochs"]
                    lr = dcfg["train_args"]["lr"]
                    ls = dcfg["train_args"].get("label_smoothing", 0.0)
                    model = train_dl_model(model, binary_matrix, t, seq_len=sl, epochs=ep, lr=lr, label_smoothing=ls)
                    dl_models[name] = (model, sl)
                except:
                    pass

            try:
                struct_predictor.train(struct_df, t)
            except:
                pass

        # ── 예측 ──
        xgb_probs = np.ones(45) / 45
        if xgb_models and t >= 20:
            try:
                seq = binary_matrix[t - 20:t].flatten().reshape(1, -1)
                xgb_probs = np.array([m.predict_proba(seq)[0, 1] for m in xgb_models])
            except:
                pass

        dl_probs_sum = np.zeros(45)
        for name in top_models:
            if dl_models[name] is not None:
                model, sl = dl_models[name]
                p = predict_dl_model(model, binary_matrix, t, seq_len=sl)
                if p is not None:
                    dl_probs_sum += p
                else:
                    dl_probs_sum += np.ones(45) / 45
            else:
                dl_probs_sum += np.ones(45) / 45

        scorer_probs = scorer.score_numbers(df, t, current_regime)

        combined = (w_xgb * xgb_probs +
                    (0.45) * (dl_probs_sum / max(n_dl, 1)) +
                    w_scorer * scorer_probs)
        combined = combined / combined.sum()

        struct_pred = struct_predictor.predict(struct_df, t) if struct_predictor.models else {}
        adjusted, cold_pool = apply_adjustments_v(combined, df.iloc[:t], t, cfg)
        pool = select_pool_v(adjusted, cold_pool, cfg)

        prev_nums = None
        bt_streaks = None
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
            games = partition_v(pool, struct_pred, cfg, None, prev_nums, bt_streaks)
        else:
            p = list(pool)
            np.random.shuffle(p)
            games = [sorted(p[i*6:(i+1)*6]) for i in range(5)]

        actual = set()
        for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
            actual.add(int(df.iloc[t][col]))

        pool_hits = len(actual & set(pool))
        game_hits = [len(actual & set(g)) for g in games]

        results.append({
            "pool_hits": pool_hits,
            "best_game": max(game_hits),
            "game_hits": game_hits,
        })

    pool_all = [r["pool_hits"] for r in results]
    best_all = [r["best_game"] for r in results]
    n = len(results)
    grade5 = sum(1 for r in results if max(r["game_hits"]) >= 3)
    grade4 = sum(1 for r in results if max(r["game_hits"]) >= 4)
    grade5_hit = sum(1 for r in results if max(r["game_hits"]) == 5)

    return {
        "pool_avg": np.mean(pool_all),
        "best_avg": np.mean(best_all),
        "grade5": grade5,
        "grade4": grade4,
        "grade5_hit": grade5_hit,
        "n": n,
    }


# ════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("  고급 딥러닝 모델 실험 (LSTM/GRU/Attention/DeepTF/ConvTF)")
    print(f"  범위: 800~1223 (스크리닝)")
    print("=" * 80)

    # Phase 1: 각 DL 모델 단독 성능 (pool hit rate)
    print("\n  ▶ Phase 1: DL 모델 단독 성능")
    dl_results = {}

    for name in DL_CONFIGS:
        dcfg = DL_CONFIGS[name]
        print(f"    [{name}] ...", end="", flush=True)
        t0 = time.time()
        r = dl_backtest(name, dcfg, train_start=800, retrain_interval=50)
        elapsed = time.time() - t0
        dl_results[name] = r
        print(f"  pool={r['pool_avg']:.3f}/6  ({elapsed:.0f}s)")

    # 순위
    ranked = sorted(dl_results.items(), key=lambda x: x[1]["pool_avg"], reverse=True)
    print("\n  DL 모델 순위 (pool hit rate):")
    for i, (name, r) in enumerate(ranked, 1):
        mark = " *" if i <= 3 else ""
        print(f"    {i:>2}. {name:<20} pool={r['pool_avg']:.3f}/6{mark}")

    # Phase 2: 상위 3개 DL 모델로 앙상블 백테스트
    top3 = [name for name, _ in ranked[:3]]
    print(f"\n  ▶ Phase 2: 상위 3개 앙상블 ({', '.join(top3)})")
    print(f"    실행 중...", end="", flush=True)
    t0 = time.time()
    ens_result = dl_ensemble_backtest(top3, train_start=800, retrain_interval=50)
    elapsed = time.time() - t0
    print(f"  pool={ens_result['pool_avg']:.3f}  best={ens_result['best_avg']:.3f}  "
          f"5등={ens_result['grade5']}  4등+={ens_result['grade4']}  ({elapsed:.0f}s)")

    # 결과 저장
    save_data = {
        "dl_individual": {n: r for n, r in dl_results.items()},
        "top3_models": top3,
        "dl_ensemble": ens_result,
    }
    with open("dl_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n  결과 저장: dl_experiment_results.json")
