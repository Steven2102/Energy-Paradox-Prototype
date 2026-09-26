import re, glob, json, csv, subprocess
from datetime import datetime

PAT_PERIOD = re.compile(r"Billing Period:\s*(\d{2}-\w{3}-\d{4})\s*to\s*(\d{2}-\w{3}-\d{4})\s*\((\d+) days\)\s*,Total Usage:\s*([\d.]+) kWh")
PAT_TOTAL  = re.compile(r"^TOTAL \(including \$([\d,.]+) GST\)\s+\$([\d,.]+)", re.M)
LABELS = ["Daily Charge", "Peak Usage", "Offpeak Usage", "Shoulder Usage", "Controlled Load"]
PAT_LINE = re.compile(
    r"^\s*(" + "|".join(LABELS) + r")\s+(\d{2}-\w{3})\s*-\s*(\d{2}-\w{3})\s+"
    r"([\d,]+\.?\d*)\s*(kWh|Days)\s+x \$([\d.]+)\s+\$([\d,.]+)", re.M)

ROWS = []
for path in sorted(glob.glob("*.pdf")):
    txt = subprocess.run(["pdftotext", "-layout", path, "-"],
                         capture_output=True, text=True).stdout
    m = PAT_PERIOD.search(txt)
    if not m:
        print("NO PERIOD:", path); continue
    start, end, days, total = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))

    # Charge lines appear twice (summary page, then detail page). Take the first
    # contiguous run -- but that run may hold SEVERAL rate blocks when a price
    # change falls inside the billing period. Key by (label, sub-period).
    blocks = {}
    for lm in PAT_LINE.finditer(txt):
        label, p0, p1, qty, unit, rate, amt = lm.groups()
        key = (label, p0, p1)
        if key in blocks:
            continue                                  # detail page repeat
        blocks[key] = (float(qty.replace(",", "")), float(rate),
                       float(amt.replace(",", "")))

    rec = {"file": path, "start": start, "end": end, "days": days,
           "stated_total_kwh": total, "n_rate_blocks": len({k[1:] for k in blocks})}

    for label in LABELS:
        parts = [(q, r, a) for (l, _, _), (q, r, a) in blocks.items() if l == label]
        k = label.lower().replace(" ", "_")
        rec[f"{k}_qty"]  = round(sum(p[0] for p in parts), 2)
        rec[f"{k}_amt"]  = round(sum(p[2] for p in parts), 2)
        rec[f"{k}_rate"] = max(p[1] for p in parts) if parts else 0.0   # latest rate

    rec["sum_kwh"] = round(rec["peak_usage_qty"] + rec["offpeak_usage_qty"]
                           + rec["shoulder_usage_qty"] + rec["controlled_load_qty"], 2)
    rec["reconciles"] = abs(rec["sum_kwh"] - total) < 0.05

    t = PAT_TOTAL.search(txt)
    if t:
        rec["gst"] = float(t.group(1).replace(",", ""))
        rec["total_aud"] = float(t.group(2).replace(",", ""))
    rec["has_solar"] = bool(re.search(r"Feed.?in Tariff|Solar Export\s+\S+\s+[\d.]+ kWh", txt, re.I))
    rec["start_dt"] = datetime.strptime(start, "%d-%b-%Y").date().isoformat()
    ROWS.append(rec)

json.dump(ROWS, open("bills.json", "w"), indent=1)
cols = ["start_dt", "start", "end", "days", "stated_total_kwh", "sum_kwh", "reconciles",
        "n_rate_blocks", "peak_usage_qty", "offpeak_usage_qty", "shoulder_usage_qty",
        "controlled_load_qty", "peak_usage_rate", "offpeak_usage_rate",
        "shoulder_usage_rate", "controlled_load_rate", "daily_charge_rate", "total_aud"]
with open("bills.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in ROWS: w.writerow(r)

print(f"{len(ROWS)} bills parsed, "
      f"{sum(1 for r in ROWS if r['reconciles'])} reconcile against stated total\n")
hdr = f"{'period':<24}{'d':>3}{'kWh':>9}{'peak':>8}{'offpk':>8}{'shldr':>8}{'ctrl':>8}{'blk':>4}{'$':>9}  ok"
print(hdr); print("-" * len(hdr))
for r in ROWS:
    print(f"{r['start'][:9]}->{r['end'][:9]:<10}{r['days']:>3}{r['stated_total_kwh']:>9.1f}"
          f"{r['peak_usage_qty']:>8.1f}{r['offpeak_usage_qty']:>8.1f}"
          f"{r['shoulder_usage_qty']:>8.1f}{r['controlled_load_qty']:>8.1f}"
          f"{r['n_rate_blocks']:>4}{r.get('total_aud',0):>9.2f}  {'Y' if r['reconciles'] else 'N'}")
