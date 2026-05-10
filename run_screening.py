"""스크리닝 실행 (800~1223회차)"""
import json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import fast_backtest, EXPERIMENTS

selected = [
    'A_baseline_v1', 'B_current_v2', 'C_soft_only',
    'D_xgb_heavy', 'E_tf_heavy', 'F_scorer_heavy',
    'G_xgb_deep', 'H_tf_big', 'I_struct_strong',
    'J_pool35', 'K_no_cold', 'L_best_candidate',
]

print('=' * 80)
print('  스크리닝 (800~1223, ~423회)')
print('=' * 80)

results = {}
for name in selected:
    cfg = EXPERIMENTS[name]
    print(f'  [{name}] ...', end='', flush=True)
    r = fast_backtest(cfg, train_start=800, retrain_interval=50)
    results[name] = r
    pct = r['grade5'] / r['n'] * 100
    print(f'  pool={r["pool_avg"]:.3f}  best={r["best_avg"]:.3f}  '
          f'5등={r["grade5"]}({pct:.1f}%)  4등+={r["grade4"]}  '
          f'5개={r["grade5_hit"]}  ({r["elapsed"]:.0f}s)')

print()
ranked = sorted(results.items(),
                key=lambda x: (x[1]['grade4'], x[1]['grade5'], x[1]['best_avg']),
                reverse=True)
print('순위표:')
for i, (name, r) in enumerate(ranked, 1):
    mark = ' *' if i <= 3 else ''
    pct = r['grade5'] / r['n'] * 100
    print(f'  {i:>2}. {name:<25} pool={r["pool_avg"]:.3f} '
          f'best={r["best_avg"]:.3f} 5등={r["grade5"]:>3} '
          f'4등+={r["grade4"]:>2} 5개={r["grade5_hit"]}{mark}')

with open('screening_results.json', 'w', encoding='utf-8') as f:
    save = {}
    for n, r in results.items():
        save[n] = {**r, 'config': EXPERIMENTS[n]}
    json.dump(save, f, indent=2, ensure_ascii=False)
print('\n저장: screening_results.json')
