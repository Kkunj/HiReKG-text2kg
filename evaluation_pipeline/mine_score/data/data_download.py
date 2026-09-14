import pandas as pd

# # Load parquet from HF
# df = pd.read_parquet(
#     "hf://datasets/kyssen/kg-gen-evaluation-answers/data/train-00000-of-00001.parquet"
# )

# # Save as JSON (records format is usually what you want)
# df.to_json("kg_gen_eval.json", orient="records", lines=True)

import pandas as pd

df = pd.read_parquet("hf://datasets/kyssen/kg-gen-evaluation-essays/data/train-00000-of-00001.parquet")

df.to_json("kg_gen_eval_essays.json", orient="records", lines=True)