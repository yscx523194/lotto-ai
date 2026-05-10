"""
로또 6/45 안 나오는 패턴 분석
==============================
1222회 실제 데이터에서 극히 드문/불가능 패턴을 발굴.
→ 필터에 반영하여 예측 정확도 향상.
"""

import json
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터 로딩
with open(os.path.join(BASE_DIR, "lotto_cache.json"), "r", encoding="utf-8") as f:
    cache = json.load(f)

rounds = []
for val in cache.values():
    nums = sorted(val["당첨번호"])
    rounds.append(nums)
rounds_arr = np.array(rounds)
TOTAL = len(rounds)

print("=" * 70)
print(f"  로또 6/45 '안 나오는 패턴' 분석 ({TOTAL}회 데이터)")
print("=" * 70)


# ════════════════════════════════════════════════════════════
# 1. 합계 범위
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  1. 합계 분포")
print(f"{'─' * 70}")

sums = [sum(r) for r in rounds]
print(f"  최소: {min(sums)}  최대: {max(sums)}  평균: {np.mean(sums):.1f}  표준편차: {np.std(sums):.1f}")

# 구간별
ranges = [(21,80), (81,90), (91,100), (101,120), (121,140), (141,160),
          (161,180), (181,200), (201,210), (211,255)]
for lo, hi in ranges:
    cnt = sum(1 for s in sums if lo <= s <= hi)
    pct = cnt / TOTAL * 100
    bar = "█" * int(pct)
    print(f"  {lo:>3}~{hi:<3}: {cnt:>4}회 ({pct:>5.1f}%) {bar}")


# ════════════════════════════════════════════════════════════
# 2. 홀짝 분포
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  2. 홀짝 비율")
print(f"{'─' * 70}")

odd_counts = [sum(1 for n in r if n % 2 == 1) for r in rounds]
for oc in range(7):
    cnt = odd_counts.count(oc)
    pct = cnt / TOTAL * 100
    bar = "█" * int(pct)
    print(f"  홀{oc}짝{6-oc}: {cnt:>4}회 ({pct:>5.1f}%) {bar}")


# ════════════════════════════════════════════════════════════
# 3. 연번 (연속 숫자) 패턴
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  3. 연번 (연속 숫자) 패턴")
print(f"{'─' * 70}")

def count_consecutive_pairs(nums):
    return sum(1 for i in range(5) if nums[i+1] - nums[i] == 1)

def has_n_consecutive(nums, n):
    for i in range(7 - n):
        if all(nums[i+j+1] - nums[i+j] == 1 for j in range(n - 1)):
            return True
    return False

consec_pairs = [count_consecutive_pairs(r) for r in rounds]
for cp in range(6):
    cnt = consec_pairs.count(cp)
    pct = cnt / TOTAL * 100
    print(f"  연번쌍 {cp}개: {cnt:>4}회 ({pct:>5.1f}%)")

print()
for n in [2, 3, 4, 5, 6]:
    cnt = sum(1 for r in rounds if has_n_consecutive(r, n))
    pct = cnt / TOTAL * 100
    print(f"  {n}연번 포함: {cnt:>4}회 ({pct:>5.1f}%)")


# ════════════════════════════════════════════════════════════
# 4. AC값 (Arithmetic Complexity)
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  4. AC값 분포")
print(f"{'─' * 70}")

def calc_ac(nums):
    diffs = set()
    for i in range(6):
        for j in range(i+1, 6):
            diffs.add(abs(nums[j] - nums[i]))
    return len(diffs) - 5

ac_values = [calc_ac(r) for r in rounds]
for ac in range(11):
    cnt = ac_values.count(ac)
    pct = cnt / TOTAL * 100
    if cnt > 0:
        bar = "█" * max(1, int(pct))
        print(f"  AC={ac:>2}: {cnt:>4}회 ({pct:>5.1f}%) {bar}")


# ════════════════════════════════════════════════════════════
# 5. 끝수 패턴
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  5. 끝수 (일의 자리) 패턴")
print(f"{'─' * 70}")

def max_same_ending(nums):
    endings = [n % 10 for n in nums]
    return max(Counter(endings).values())

def unique_endings_count(nums):
    return len(set(n % 10 for n in nums))

max_endings = [max_same_ending(r) for r in rounds]
for me in range(1, 7):
    cnt = max_endings.count(me)
    pct = cnt / TOTAL * 100
    if cnt > 0:
        print(f"  같은끝수 최대 {me}개: {cnt:>4}회 ({pct:>5.1f}%)")

print()
unique_end = [unique_endings_count(r) for r in rounds]
for ue in range(1, 7):
    cnt = unique_end.count(ue)
    pct = cnt / TOTAL * 100
    if cnt > 0:
        print(f"  서로 다른 끝수 {ue}개: {cnt:>4}회 ({pct:>5.1f}%)")


# ════════════════════════════════════════════════════════════
# 6. 10단위 구간 (decade) 분포
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  6. 10단위 구간 분포")
print(f"{'─' * 70}")

def decade_dist(nums):
    d = [0]*5
    for n in nums:
        if n <= 9: d[0] += 1
        elif n <= 19: d[1] += 1
        elif n <= 29: d[2] += 1
        elif n <= 39: d[3] += 1
        else: d[4] += 1
    return tuple(d)

decade_counter = Counter(decade_dist(r) for r in rounds)
# 빈 구간 (0개인 decade) 분석
empty_decades = [sum(1 for x in decade_dist(r) if x == 0) for r in rounds]
for ed in range(5):
    cnt = empty_decades.count(ed)
    pct = cnt / TOTAL * 100
    if cnt > 0:
        print(f"  빈 구간 {ed}개: {cnt:>4}회 ({pct:>5.1f}%)")

# 한 구간에 4개 이상 집중
max_in_decade = [max(decade_dist(r)) for r in rounds]
print()
for md in range(1, 7):
    cnt = max_in_decade.count(md)
    pct = cnt / TOTAL * 100
    if cnt > 0:
        print(f"  한 구간 최대 {md}개: {cnt:>4}회 ({pct:>5.1f}%)")


# ════════════════════════════════════════════════════════════
# 7. 소수 개수
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  7. 소수 개수")
print(f"{'─' * 70}")

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
prime_counts = [sum(1 for n in r if n in PRIMES) for r in rounds]
for pc in range(7):
    cnt = prime_counts.count(pc)
    pct = cnt / TOTAL * 100
    if cnt > 0:
        print(f"  소수 {pc}개: {cnt:>4}회 ({pct:>5.1f}%)")


# ════════════════════════════════════════════════════════════
# 8. 고저 비율 (23 기준)
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  8. 고저 비율 (23 이상 = 고)")
print(f"{'─' * 70}")

high_counts = [sum(1 for n in r if n >= 23) for r in rounds]
for hc in range(7):
    cnt = high_counts.count(hc)
    pct = cnt / TOTAL * 100
    bar = "█" * int(pct)
    print(f"  고{hc}저{6-hc}: {cnt:>4}회 ({pct:>5.1f}%) {bar}")


# ════════════════════════════════════════════════════════════
# 9. 번호 간격 (gap) 분석
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  9. 번호 간격 (gap) 분석")
print(f"{'─' * 70}")

max_gaps = [max(r[i+1] - r[i] for i in range(5)) for r in rounds]
min_gaps = [min(r[i+1] - r[i] for i in range(5)) for r in rounds]
spans = [r[5] - r[0] for r in rounds]

print(f"  최대 간격 - 평균: {np.mean(max_gaps):.1f}  최소: {min(max_gaps)}  최대: {max(max_gaps)}")
print(f"  최소 간격 - 평균: {np.mean(min_gaps):.1f}  최소: {min(min_gaps)}  최대: {max(min_gaps)}")
print(f"  스팬(n6-n1) - 평균: {np.mean(spans):.1f}  최소: {min(spans)}  최대: {max(spans)}")

print(f"\n  스팬 분포:")
span_ranges = [(5,15), (16,20), (21,25), (26,30), (31,35), (36,40), (41,44)]
for lo, hi in span_ranges:
    cnt = sum(1 for s in spans if lo <= s <= hi)
    pct = cnt / TOTAL * 100
    bar = "█" * int(pct)
    print(f"  {lo:>2}~{hi:<2}: {cnt:>4}회 ({pct:>5.1f}%) {bar}")


# ════════════════════════════════════════════════════════════
# 10. 번호 쌍 동시 출현 분석
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  10. 번호 쌍 동시 출현 (잘 안 나오는 쌍)")
print(f"{'─' * 70}")

pair_count = defaultdict(int)
for r in rounds:
    for i in range(6):
        for j in range(i+1, 6):
            pair_count[(r[i], r[j])] += 1

# 기대 동시 출현 횟수
expected_pair = TOTAL * (5/44)  # ~138.9회 중 한 쌍이 동시에 나올 기대

# 한번도 안 나온 쌍
total_pairs = 45 * 44 // 2  # 990
zero_pairs = sum(1 for p in combinations(range(1,46), 2) if pair_count.get(p, 0) == 0)
print(f"  전체 쌍: {total_pairs}개")
print(f"  한번도 동시 출현 안 한 쌍: {zero_pairs}개 ({zero_pairs/total_pairs*100:.1f}%)")
print(f"  기대 동시 출현: {expected_pair:.1f}회")

# 과다 출현 쌍 (기대의 2배 이상)
hot_pairs = [(p, c) for p, c in pair_count.items() if c >= expected_pair * 1.5]
hot_pairs.sort(key=lambda x: -x[1])
print(f"\n  과다 출현 (기대의 1.5배+): {len(hot_pairs)}쌍")
for (a, b), c in hot_pairs[:10]:
    print(f"    ({a:>2}, {b:>2}): {c}회 (기대: {expected_pair:.0f})")


# ════════════════════════════════════════════════════════════
# 11. 3의 배수 개수
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  11. 3의 배수 개수")
print(f"{'─' * 70}")

mult3 = [sum(1 for n in r if n % 3 == 0) for r in rounds]
for m in range(7):
    cnt = mult3.count(m)
    pct = cnt / TOTAL * 100
    if cnt > 0:
        print(f"  3의배수 {m}개: {cnt:>4}회 ({pct:>5.1f}%)")


# ════════════════════════════════════════════════════════════
# 12. 이전 회차 번호 재출현 분석
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  12. 이전 회차 번호 재출현")
print(f"{'─' * 70}")

carry_counts = []
for i in range(1, TOTAL):
    prev = set(rounds[i-1])
    curr = set(rounds[i])
    carry = len(prev & curr)
    carry_counts.append(carry)

for cc in range(7):
    cnt = carry_counts.count(cc)
    pct = cnt / len(carry_counts) * 100
    print(f"  이전회차에서 {cc}개 재출현: {cnt:>4}회 ({pct:>5.1f}%)")


# ════════════════════════════════════════════════════════════
# 13. 번호 합의 홀짝
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  13. 합계 홀짝")
print(f"{'─' * 70}")

sum_odd = sum(1 for s in sums if s % 2 == 1)
sum_even = TOTAL - sum_odd
print(f"  합 홀수: {sum_odd}회 ({sum_odd/TOTAL*100:.1f}%)")
print(f"  합 짝수: {sum_even}회 ({sum_even/TOTAL*100:.1f}%)")


# ════════════════════════════════════════════════════════════
# 14. 최소/최대 번호 분포
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  14. 최소/최대 번호")
print(f"{'─' * 70}")

min_nums = [r[0] for r in rounds]
max_nums = [r[5] for r in rounds]

print(f"  최소번호 - 평균: {np.mean(min_nums):.1f}  1~5이내: {sum(1 for m in min_nums if m <= 5)/TOTAL*100:.1f}%  10+: {sum(1 for m in min_nums if m >= 10)/TOTAL*100:.1f}%")
print(f"  최대번호 - 평균: {np.mean(max_nums):.1f}  40+: {sum(1 for m in max_nums if m >= 40)/TOTAL*100:.1f}%  35이하: {sum(1 for m in max_nums if m <= 35)/TOTAL*100:.1f}%")


# ════════════════════════════════════════════════════════════
# 15. 복합 패턴: 거의 안 나오는 조합
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  15. 복합 '안 나오는 패턴' 요약 (하드필터 후보)")
print(f"{'─' * 70}")

filters = []

# 합계
f = sum(1 for s in sums if s < 90 or s > 200)
filters.append(("합계 < 90 or > 200", f, f/TOTAL*100))

# 홀짝 극단
f = sum(1 for oc in odd_counts if oc == 0 or oc == 6)
filters.append(("올홀 or 올짝", f, f/TOTAL*100))

# AC <= 4
f = sum(1 for ac in ac_values if ac <= 4)
filters.append(("AC ≤ 4", f, f/TOTAL*100))

# 같은 끝수 4+
f = sum(1 for me in max_endings if me >= 4)
filters.append(("같은끝수 4개+", f, f/TOTAL*100))

# 소수 5+
f = sum(1 for pc in prime_counts if pc >= 5)
filters.append(("소수 5개+", f, f/TOTAL*100))

# 3연번
f = sum(1 for r in rounds if has_n_consecutive(r, 3))
filters.append(("3연번 포함", f, f/TOTAL*100))

# 4연번
f = sum(1 for r in rounds if has_n_consecutive(r, 4))
filters.append(("4연번 포함", f, f/TOTAL*100))

# 스팬 < 20
f = sum(1 for s in spans if s < 20)
filters.append(("스팬 < 20", f, f/TOTAL*100))

# 한 구간 4개+
f = sum(1 for md in max_in_decade if md >= 4)
filters.append(("한 구간 4개+", f, f/TOTAL*100))

# 고저 극단
f = sum(1 for hc in high_counts if hc == 0 or hc == 6)
filters.append(("올고 or 올저", f, f/TOTAL*100))

# 홀짝 1:5 or 5:1
f = sum(1 for oc in odd_counts if oc == 1 or oc == 5)
filters.append(("홀1짝5 or 홀5짝1", f, f/TOTAL*100))

# 3의배수 0개 or 5+
f = sum(1 for m in mult3 if m == 0 or m >= 5)
filters.append(("3배수 0개 or 5+", f, f/TOTAL*100))

# 이전 회차 4개+ 재출현
f = sum(1 for cc in carry_counts if cc >= 4)
filters.append(("이전회차 4개+ 재출현", f, f/TOTAL*100))

# 서로 다른 끝수 3개 이하
f = sum(1 for ue in unique_end if ue <= 3)
filters.append(("서로 다른 끝수 ≤ 3", f, f/TOTAL*100))

# 최소번호 15+
f = sum(1 for m in min_nums if m >= 15)
filters.append(("최소번호 ≥ 15", f, f/TOTAL*100))

# 최대번호 ≤ 35
f = sum(1 for m in max_nums if m <= 35)
filters.append(("최대번호 ≤ 35", f, f/TOTAL*100))

# 합계 100~170 밖 (좀 더 타이트)
f = sum(1 for s in sums if s < 100 or s > 170)
filters.append(("합계 < 100 or > 170", f, f/TOTAL*100))

# 정렬: 빈도 순
filters.sort(key=lambda x: x[1])

print(f"\n  {'패턴':<25} {'출현':>6} {'비율':>7}  강도")
print(f"  {'─'*55}")
for name, cnt, pct in filters:
    if pct < 1:
        strength = "🔴 극히 드묾 (하드필터)"
    elif pct < 5:
        strength = "🟠 매우 드묾 (하드필터)"
    elif pct < 10:
        strength = "🟡 드묾 (소프트필터)"
    else:
        strength = "⚪ 보통 (감점)"
    print(f"  {name:<25} {cnt:>4}회 ({pct:>5.1f}%)  {strength}")


# ════════════════════════════════════════════════════════════
# 16. 현재 필터 vs 추가 가능 필터
# ════════════════════════════════════════════════════════════
print(f"\n{'─' * 70}")
print("  16. 신규 필터 추천")
print(f"{'─' * 70}")

print("""
  현재 하드 필터:
    ✅ 합계 < 90 or > 200
    ✅ 올홀/올짝
    ✅ AC ≤ 4
    ✅ 같은끝수 4개+
    ✅ 소수 5개+

  현재 소프트 필터:
    ✅ 3연번 (-1.5)
    ✅ 홀1짝5/홀5짝1 (-0.5)
    ✅ 연번쌍 3개+ (-1.0)

  🆕 추가 추천 (분석 기반):
""")

new_filters = [
    ("하드", "4연번 포함", "0.5% 미만 → -100"),
    ("하드", "스팬 < 20", f"{sum(1 for s in spans if s < 20)/TOTAL*100:.1f}% → -100"),
    ("하드", "한 구간 4개+", f"{sum(1 for md in max_in_decade if md >= 4)/TOTAL*100:.1f}% → -100"),
    ("하드", "올고 or 올저", f"{sum(1 for hc in high_counts if hc==0 or hc==6)/TOTAL*100:.1f}% → -100"),
    ("하드", "최소번호 ≥ 15", f"{sum(1 for m in min_nums if m >= 15)/TOTAL*100:.1f}% → -100"),
    ("하드", "최대번호 ≤ 35", f"{sum(1 for m in max_nums if m <= 35)/TOTAL*100:.1f}% → -100"),
    ("소프트", "서로 다른 끝수 ≤ 3", f"{sum(1 for ue in unique_end if ue <= 3)/TOTAL*100:.1f}% → -1.0"),
    ("소프트", "3배수 0개 or 5+", f"{sum(1 for m in mult3 if m==0 or m>=5)/TOTAL*100:.1f}% → -0.5"),
    ("소프트", "이전 4개+ 재출현", f"{sum(1 for cc in carry_counts if cc >= 4)/TOTAL*100:.1f}% → -0.8"),
    ("소프트", "합계 < 100 or > 170 (타이트)", f"{sum(1 for s in sums if s < 100 or s > 170)/TOTAL*100:.1f}% → -0.3"),
]

for level, name, info in new_filters:
    marker = "🔴" if level == "하드" else "🟡"
    print(f"    {marker} [{level}] {name}: {info}")

print()
