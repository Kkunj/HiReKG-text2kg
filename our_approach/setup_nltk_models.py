"""
Setup script to download required NLP models for the GraphRAG pipeline.
Run this once after installing requirements.txt
"""

import subprocess
import sys


def main():
    print("=" * 60)
    print("GraphRAG Pipeline - NLP Model Setup")
    print("=" * 60)

    # Download NLTK data
    print("\n[1/2] Downloading NLTK models...")
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    print("      ✓ NLTK models downloaded")

    # Download spaCy model
    print("\n[2/2] Downloading spaCy model (en_core_web_sm)...")
    subprocess.check_call([
        sys.executable, "-m", "spacy", "download", "en_core_web_sm", "--quiet"
    ])
    print("      ✓ spaCy model downloaded")

    print("\n" + "=" * 60)
    print(" Setup complete! You can now run the pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()

