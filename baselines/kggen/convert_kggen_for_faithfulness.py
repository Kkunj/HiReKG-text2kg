"""
Convert the raw KGGen GPT-4o MINE graphs into the directory structure
expected by evaluation_pipeline/faithfulness/run_faithfulness.py.

This run is the one KGGen configuration whose graphs were produced by the
upstream kg-gen release rather than by our batch runner, so its output uses
kg-gen's own flat `N.json` layout instead of the `doc_<N>/final_output.json`
layout every other experiment in this repo uses. This script bridges the two.

Input:  experiments/kggen_mine_gpt4o_graphs/graphs/N.json
        with {entities, relations: [[s,r,o], ...], edges}
Output: experiments/kggen_mine_temp_faithfulness/doc_{N-1}/final_output.json
        with {"triples_final": [{"subject":s, "relation":r, "object":o}, ...]}

Files 1-100 map to doc_0 through doc_99 (the 100 filtered MINE docs).
Files 101+ are included but won't have matching source texts, so the
faithfulness pipeline will skip them automatically.
"""

import json
import os
from pathlib import Path

KGGEN_DIR = Path(__file__).resolve().parent  # baselines/kggen/
INPUT_DIR = KGGEN_DIR / "experiments" / "kggen_mine_gpt4o_graphs" / "graphs"
OUTPUT_DIR = KGGEN_DIR / "experiments" / "kggen_mine_temp_faithfulness"


def convert():
    converted = 0
    skipped = 0

    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.endswith(".json") or "result" in fname.lower():
            continue

        file_num = int(fname.replace(".json", ""))
        doc_id = f"doc_{file_num - 1}"  # 1-indexed file → 0-indexed doc_id

        src = INPUT_DIR / fname
        data = json.loads(src.read_text(encoding="utf-8"))

        relations = data.get("relations", [])
        if not relations:
            print(f"  SKIP {fname}: no relations")
            skipped += 1
            continue

        triples_final = [
            {"subject": r[0], "relation": r[1], "object": r[2]}
            for r in relations
            if len(r) >= 3
        ]

        doc_dir = OUTPUT_DIR / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        out_path = doc_dir / "final_output.json"
        out_path.write_text(
            json.dumps({"triples_final": triples_final}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        converted += 1
        print(f"  {fname} -> {doc_id}/final_output.json ({len(triples_final)} triples)")

    print(f"\nDone: converted={converted}, skipped={skipped}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert()
