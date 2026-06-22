"""Second-pass diagnostic to find any remaining issues."""
import json, re
from pathlib import Path
from collections import Counter

OUT = Path('/home/user/output_v2')

issues = {
    "no_title": [],
    "no_doi": [],
    "no_authors": [],
    "few_refs": [],          # <10 refs in a paper that should have many
    "no_cites_linked": [],   # 0 cites linked despite many refs
    "br_remaining": [],      # still has <br> tags (mostly OK but worth knowing)
    "junk_first_para": [],
}

papers = []
for d in sorted(OUT.iterdir()):
    if not d.is_dir():
        continue
    stats_p = d / "stats.json"
    if not stats_p.exists():
        continue
    s = json.load(open(stats_p))
    md = (d / "paper.md").read_text()
    papers.append(s)
    
    if not s.get("title"):
        issues["no_title"].append(d.name)
    if not s.get("doi"):
        issues["no_doi"].append(d.name)
    if not s.get("n_authors"):
        issues["no_authors"].append(d.name)
    if s.get("n_references", 0) < 10:
        issues["few_refs"].append((d.name, s.get("n_references", 0)))
    if s.get("n_references", 0) >= 20 and s.get("n_citations_linked", 0) == 0:
        issues["no_cites_linked"].append((d.name, s.get("n_references", 0)))
    n_br = md.count("<br>")
    if n_br > 20:
        issues["br_remaining"].append((d.name, n_br))
    
    # Junk first paragraph?
    first_section = md.split("---", 1)[-1].strip()[:500]
    if "doi:" in first_section.lower() or "received:" in first_section.lower():
        issues["junk_first_para"].append(d.name)

print(f"\n=== Summary across {len(papers)} papers ===")
print(f"  Avg refs: {sum(p.get('n_references',0) for p in papers)/len(papers):.1f}")
print(f"  Avg cites linked: {sum(p.get('n_citations_linked',0) for p in papers)/len(papers):.1f}")
print(f"  Median cov: {sorted(p['coverage_ratio'] for p in papers)[len(papers)//2]}")

for k, v in issues.items():
    print(f"\n=== {k} ({len(v)}) ===")
    for item in v[:10]:
        print(f"  {item}")
