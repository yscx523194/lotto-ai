"""
로또 6/45 즉시 예측 (운영 전용)
================================
meta.json만 읽어서 5게임 즉시 생성.
torch, xgboost 등 무거운 라이브러리 불필요.

의존성: numpy (이것 하나만!)

사용법:
    # Python에서
    from predict_fast import predict
    result = predict()
    print(result["games"])  # [[1,2,3,4,5,6], ...]

    # CLI
    python predict_fast.py
    python predict_fast.py --json   # JSON 출력
"""

import os
import json
import numpy as np
from itertools import combinations
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "lotto_models")


# ════════════════════════════════════════════════════════════
# 조합 점수 + 필터 (학습 불필요, 순수 계산)
# ════════════════════════════════════════════════════════════

def _score_combination(combo, struct_pred):
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


def _partition_into_5_games(pool_30, struct_pred, n_attempts=30, n_candidates=100):
    """30개 번호를 5게임(6개씩)으로 최적 분할 (경량 버전)"""
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
                total_score += _score_combination(tuple(combo), struct_pred)
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
                    sc = _score_combination(tuple(combo), struct_pred)
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
                    total_score += _score_combination(tuple(combo), struct_pred)
                    remaining = remaining[6:]

        if len(partition) == 5 and total_score > best_total_score:
            best_total_score = total_score
            best_partition = partition

    if best_partition is None:
        np.random.shuffle(pool)
        best_partition = [sorted(pool[i*6:(i+1)*6]) for i in range(5)]

    return best_partition


# ════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════

def predict(model_dir: str = None) -> dict:
    """
    meta.json에서 캐시된 풀/점수를 읽어 5게임 즉시 생성.
    
    Args:
        model_dir: lotto_models 디렉토리 경로 (기본: 스크립트 옆 lotto_models/)

    Returns:
        {
            "target_round": 1224,
            "games": [[1,2,3,4,5,6], ...],  # 5게임
            "pool_30": [...],
            "excluded_15": [...],
            "cold_in_pool": 15,
            "model_trained_on": 1223,
        }

    Raises:
        FileNotFoundError: meta.json이 없으면 (train_model.py 먼저 실행)
    """
    if model_dir is None:
        model_dir = MODEL_DIR

    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"모델 파일 없음: {meta_path}\n"
            f"로컬에서 train_model.py를 실행 후 git push하세요."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    pool_30 = meta["pool_30"]
    struct_pred = meta["struct_pred"]

    # 5게임 분할 (매번 다른 조합 — 같은 풀에서 새로운 파티션)
    games = _partition_into_5_games(pool_30, struct_pred)

    return {
        "target_round": meta["target_round"],
        "games": games,
        "pool_30": pool_30,
        "excluded_15": meta["excluded_15"],
        "cold_in_pool": meta["cold_in_pool"],
        "model_trained_on": meta["trained_on"],
        "scores": meta.get("scores"),
        "struct_pred": struct_pred,
    }


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import time

    t0 = time.time()
    result = predict()
    elapsed = time.time() - t0

    if "--json" in sys.argv:
        # JSON 출력 (API 서버 연동용)
        output = {
            "target_round": result["target_round"],
            "games": result["games"],
            "pool_30": result["pool_30"],
            "excluded_15": result["excluded_15"],
            "cold_in_pool": result["cold_in_pool"],
            "model_trained_on": result["model_trained_on"],
            "elapsed_ms": round(elapsed * 1000),
        }
        print(json.dumps(output, ensure_ascii=False))
    else:
        print(f"\n  예측 완료 ({elapsed:.2f}초, 모델: {result['model_trained_on']}회차 학습)")
        print(f"\n{'=' * 60}")
        print(f"  {result['target_round']}회차 예측 (최대 커버리지)")
        print(f"{'=' * 60}")
        print(f"\n  풀 30개: {result['pool_30']}")
        print(f"  제외 15개: {result['excluded_15']}")
        print(f"  cold 포함: {result['cold_in_pool']}개")
        print(f"\n  5게임:")
        for i, g in enumerate(result["games"], 1):
            s = sum(g)
            odd = sum(1 for n in g if n % 2 == 1)
            print(f"    게임{i}: {g}  합={s} 홀={odd}")
        print()
