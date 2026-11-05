import json
from pathlib import Path
root=Path(__file__).resolve().parent
expected=json.loads((root/'expected_source_fixture.json').read_text())
actual=json.loads((root/'runtime_inventory.json').read_text())
report={}
for group in ['source','fixture']:
 a=expected[group];b=actual[group]
 differences=[key for key in sorted(set(a)|set(b)) if a.get(key)!=b.get(key)]
 report[group]={'files':len(b),'mismatches':differences}
(root/'source_fixture_check.json').write_text(json.dumps(report,indent=2)+'\n')
assert all(not x['mismatches'] for x in report.values()),report
print(json.dumps(report))
