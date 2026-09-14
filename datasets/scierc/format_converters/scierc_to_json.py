"""
Convert sciERC raw data to standardized format.

Reads triplets.txt and target.txt from the processed sciERC output directory.
- triplets.txt: each line is a JSON list of [head, relation, tail] triplets.
- target.txt: each non-blank line is the source text for the corresponding triplets.

Outputs:
- triplets/doc_N.json — list of {"head", "relation", "tail"} objects
- texts/doc_N.txt     — source text for that document
"""

import json
from pathlib import Path

INPUT_DIR = Path(r"C:\<PROJECT_ROOT>\multimodal_RAG\GraphRAG\Datasets\sciERC_raw\processed_scierc\output")
TRIPLETS_FILE = INPUT_DIR / "triplets.txt"
TARGET_FILE = INPUT_DIR / "target.txt"

OUTPUT_TRIPLETS_DIR = Path(r"C:\<PROJECT_ROOT>\graph_rag\datasets\scierc\triplets")
OUTPUT_TEXTS_DIR = Path(r"C:\<PROJECT_ROOT>\graph_rag\datasets\scierc\texts")


def convert_line(raw_triplets: list[list[str]]) -> list[dict]:
    return [
        {"head": t[0], "relation": t[1], "tail": t[2]}
        for t in raw_triplets
    ]


def read_non_blank_lines(filepath: Path) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    OUTPUT_TRIPLETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    triplet_lines = read_non_blank_lines(TRIPLETS_FILE)
    text_lines = read_non_blank_lines(TARGET_FILE)

    assert len(triplet_lines) == len(text_lines), (
        f"Mismatch: {len(triplet_lines)} triplet lines vs {len(text_lines)} text lines"
    )

    for i, (triplet_line, text) in enumerate(zip(triplet_lines, text_lines)):
        raw_triplets = json.loads(triplet_line)
        triplets = convert_line(raw_triplets)

        with open(OUTPUT_TRIPLETS_DIR / f"doc_{i}.json", "w", encoding="utf-8") as out:
            json.dump(triplets, out, indent=2)

        with open(OUTPUT_TEXTS_DIR / f"doc_{i}.txt", "w", encoding="utf-8") as out:
            out.write(text)

    print(f"Created {len(triplet_lines)} triplet JSON files in {OUTPUT_TRIPLETS_DIR}")
    print(f"Created {len(text_lines)} text files in {OUTPUT_TEXTS_DIR}")


if __name__ == "__main__":
    main()
