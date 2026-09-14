import json
import os
from datasets import Dataset, DatasetDict
from huggingface_hub import whoami
from typing import List, Dict, Any


def extract_accuracy(responses: Dict[str, Any] | None) -> float | None:
    """Extract accuracy from response object and convert to float"""
    if responses is None:
        return None

    # Get the last item if it's a list, otherwise use the object directly
    if isinstance(responses, list) and len(responses) > 0:
        last_item = responses[-1]
    else:
        last_item = responses

    if isinstance(last_item, dict) and "accuracy" in last_item:
        accuracy_str = last_item["accuracy"]
        if isinstance(accuracy_str, str) and accuracy_str.endswith("%"):
            try:
                return float(accuracy_str.rstrip("%")) / 100.0
            except ValueError:
                return None
    return None


def load_kg_queries_and_essays() -> List[Dict[str, Any]]:
    """Load all KG JSON files, their corresponding generated queries, essays, and evaluation results"""

    # Load answers (generated queries)
    with open("experiments/MINE/answers.json", "r") as f:
        all_answers = json.load(f)

    # Load essays
    with open("experiments/MINE/essays.json", "r") as f:
        all_essays = json.load(f)

    # Find all {i}.json files (excluding {i}_results*.json)
    kg_files, graphrag_files, openie_files = [], [], []
    for i in range(1, 110):
        file_path = f"experiments/MINE/results/kggen/{i}.json"
        if os.path.exists(file_path):
            kg_files.append((i, file_path))
        file_path = f"experiments/MINE/results/GraphRAG/{i}.json"
        if os.path.exists(file_path):
            graphrag_files.append((i, file_path))
        file_path = f"experiments/MINE/results/OpenIE/{i}.json"
        if os.path.exists(file_path):
            openie_files.append((i, file_path))

    print(f"Found {len(kg_files)} KG files")
    print(f"Found {len(graphrag_files)} GraphRAG files")
    print(f"Found {len(openie_files)} OpenIE files")
    # Create dataset entries
    dataset_entries = []

    for idx, file_path in kg_files:
        # Load KG data
        with open(file_path, "r") as f:
            kg_data = json.load(f)

        # Load KG responses data
        kg_responses_path = file_path.replace(".json", "_results.json")
        if not os.path.exists(kg_responses_path):
            # Try alternative suffixes
            kg_responses_path = file_path.replace(".json", "_resultsG.json")
            if not os.path.exists(kg_responses_path):
                kg_responses_path = file_path.replace(".json", "_resultsN.json")
        if os.path.exists(kg_responses_path):
            with open(kg_responses_path, "r") as f:
                kg_responses = json.load(f)
        else:
            kg_responses = None

        graphrag_file_path = file_path.replace("kggen", "GraphRAG")
        if os.path.exists(graphrag_file_path):
            with open(graphrag_file_path, "r") as f:
                graphrag_data = json.load(f)
        else:
            graphrag_data = None

        # Load GraphRAG responses data
        graphrag_responses_path = graphrag_file_path.replace(".json", "_resultsG.json")
        if os.path.exists(graphrag_responses_path):
            with open(graphrag_responses_path, "r") as f:
                graphrag_responses = json.load(f)
        else:
            graphrag_responses = None

        openie_file_path = file_path.replace("kggen", "OpenIE")
        if os.path.exists(openie_file_path):
            with open(openie_file_path, "r") as f:
                openie_data = json.load(f)
        else:
            openie_data = None

        # Load OpenIE responses data
        openie_responses_path = openie_file_path.replace(".json", "_resultsST.json")
        if os.path.exists(openie_responses_path):
            with open(openie_responses_path, "r") as f:
                openie_responses = json.load(f)
        else:
            openie_responses = None

        generated_queries = all_answers[idx - 1] if idx - 1 < len(all_answers) else []
        generated_queries = [qa["answer"] for qa in generated_queries]

        essay = (
            all_essays[idx - 1]
            if idx - 1 < len(all_essays)
            else {"topic": "", "content": ""}
        )

        entry = {
            "id": idx,
            "essay_topic": essay.get("topic", ""),
            "essay_content": essay.get("content", ""),
            "generated_queries": generated_queries,
            "num_generated_queries": len(generated_queries),
            "kggen": kg_data,
            "graphrag_kg": graphrag_data,
            "openie_kg": openie_data,
            "kggen_responses": kg_responses,
            "graphrag_responses": graphrag_responses,
            "openie_responses": openie_responses,
            "kggen_accuracy": extract_accuracy(kg_responses),
            "graphrag_accuracy": extract_accuracy(graphrag_responses),
            "openie_accuracy": extract_accuracy(openie_responses),
        }

        dataset_entries.append(entry)
        print(
            f"Processed {file_path} - Generated queries: {entry['num_generated_queries']}, "
            f"Responses: KG={kg_responses is not None}, GraphRAG={graphrag_responses is not None}, OpenIE={openie_responses is not None}, "
            f"Accuracies: KG={entry['kggen_accuracy']}, GraphRAG={entry['graphrag_accuracy']}, OpenIE={entry['openie_accuracy']}"
        )

    return dataset_entries


def create_and_upload_dataset(repo_name: str = "kg-gen-MINE-evaluation-dataset"):
    """Create a Hugging Face dataset and optionally upload it"""

    # Get authenticated user and append to repo name if not already included
    if "/" not in repo_name:
        user_info = whoami()
        repo_name = f"{user_info['name']}/{repo_name}"

    # Load data
    print("Loading KG files, generated queries, and essays...")
    dataset_entries = load_kg_queries_and_essays()
    dataset = Dataset.from_list(dataset_entries)
    dataset_dict = DatasetDict({"train": dataset})

    print(f"\nUploading dataset to {repo_name}...")
    dataset_dict.push_to_hub(
        repo_name,
        private=False,
        commit_message="Upload KG evaluation dataset with generated queries and essays",
    )

    print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_name}")
    return dataset_dict


def verify_dataset(repo_name="kg-gen-MINE-evaluation-dataset"):
    """Verify the uploaded dataset by loading it back"""
    from datasets import load_dataset

    if "/" not in repo_name:
        user_info = whoami()
        repo_name = f"{user_info['name']}/{repo_name}"

    print(f"\nVerifying dataset from {repo_name}...")
    dataset = load_dataset(repo_name)

    print("Dataset loaded successfully!")
    print(f"Number of entries: {len(dataset['train'])}")

    # Show first entry as example
    if len(dataset["train"]) > 0:
        first_entry = dataset["train"][0]
        print("\nFirst entry example:")
        print(f"ID: {first_entry['id']}")
        print(f"Essay topic: {first_entry['essay_topic'][:50]}...")
        print(f"Essay content length: {len(first_entry['essay_content'])} characters")
        print(f"Number of generated queries: {len(first_entry['generated_queries'])}")


def main():
    """Main function with options"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload KG evaluation dataset to Hugging Face"
    )
    parser.add_argument(
        "--repo-name",
        default="kg-gen-MINE-evaluation-dataset",
        help="Hugging Face repository name",
    )

    args = parser.parse_args()
    create_and_upload_dataset(repo_name=args.repo_name)
    verify_dataset(args.repo_name)


if __name__ == "__main__":
    main()
