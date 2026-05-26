import csv
from pathlib import Path
p = Path('data/resumes_cleaned.csv')
if not p.exists():
    print('ERROR: data/resumes_cleaned.csv not found')
    raise SystemExit(1)

cats = set()
with p.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        cat = (r.get('category') or '').strip()
        if cat:
            cats.add(cat)

for c in sorted(cats):
    print(c)
