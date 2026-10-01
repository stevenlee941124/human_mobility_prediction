"""
===============================================================================
HuMob 2026: Step 2 - Compute SOTA v3 Per-Route Physical Macro Baseline
===============================================================================
Executes the SOTA v3 physical macro baseline framework:
1. 隔離 7,976 條全零/無人區路線 (Class 1: Persistent Zero, 52.7%)。
2. 提取每週波峰波谷中點（Midpoint: 0.5 * (max + min)）作為純淨中軸錨點。
3. 結合一月份 7 日滾動滑動平均過渡（震後衝擊響應）。
4. 預測四月平衡容納量 K = mean(April anchors)。
5. 60 天盲區指數阻尼微分方程過渡橋樑。
6. 366 天全域 1D 插值與高斯平滑（sigma=2.0）達成 C^1 一階連續。
===============================================================================
"""
import sys
import time
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT  = PIPELINE_ROOT.parent

PROCESSED = PROJECT_ROOT / 'data' / 'processed'
OUT_DIR = PROJECT_ROOT / 'data' / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PIPELINE_ROOT / 'src'))
from per_route_sota_v3_baseline import build_all_v3_baselines

OD_PKL    = PROCESSED / 'od_time_series.pkl'
DATES_PKL = PROCESSED / 'dates.pkl'
DESTS_PKL = PROCESSED / 'eval_destinations.pkl'

OUT_BASE_BACKUP = OUT_DIR / 'per_route_sota_v3_baseline_backup.pkl'
OUT_META_V1     = OUT_DIR / 'per_route_exponential_baseline_meta_v1_sota.pkl'

print("=" * 80)
print("🚀 [Step 2] Computing SOTA v3 Physical Macro Baseline (9 Disaster Classes)")
print("=" * 80)
t0 = time.time()

with open(OD_PKL, 'rb') as f: od_ts = pickle.load(f)
with open(DATES_PKL, 'rb') as f: dates_str = pickle.load(f)
with open(DESTS_PKL, 'rb') as f: eval_destinations = pickle.load(f)

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

print(f"📖 Loaded {len(od_ts):,} OD routes. Building SOTA v3 macro baselines...")
baselines, meta_out = build_all_v3_baselines(
    od_ts=od_ts,
    dates_str=dates_str,
    cal_dates=cal_dates,
    cal_date_to_idx=cal_date_to_idx
)

meta_out['eval_destinations'] = eval_destinations

# Save baseline outputs
with open(OUT_BASE_BACKUP, 'wb') as f:
    pickle.dump(baselines, f)
print(f"💾 Macro Baseline Backup saved: {OUT_BASE_BACKUP.name} ({OUT_BASE_BACKUP.stat().st_size / (1024*1024):.2f} MB)")

with open(OUT_META_V1, 'wb') as f:
    pickle.dump(meta_out, f)
print(f"💾 Macro Baseline Meta saved: {OUT_META_V1.name} ({OUT_META_V1.stat().st_size / 1024:.2f} KB)")

# Class summary
route_info = meta_out.get('route_info', {})
class_counts = {}
for info in route_info.values():
    cid = info.get('class_id', 0)
    cname = info.get('class_name', 'Unknown')
    class_counts[(cid, cname)] = class_counts.get((cid, cname), 0) + 1

print("\n📊 9-Class Distribution Summary:")
for (cid, cname), cnt in sorted(class_counts.items()):
    print(f"   Class {cid:2d} ({cname}): {cnt:5,d} routes ({cnt/len(od_ts)*100:.1f}%)")

print(f"\n✅ [Step 2] Completed in {time.time() - t0:.2f} seconds!")
