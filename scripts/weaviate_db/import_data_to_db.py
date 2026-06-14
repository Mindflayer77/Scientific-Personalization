import gc
import json
import os

import pandas as pd
import weaviate
from tqdm import tqdm
from utils.config import CONFIG

RAW_BASE_PATH = CONFIG['paths']['base_path']
ABS_BASE_PATH = os.path.expanduser(RAW_BASE_PATH)

CHUNKS_DIR  = os.path.join(ABS_BASE_PATH, CONFIG['paths']['articles_dir'])

client = weaviate.connect_to_local()
collection = client.collections.get("ResearchPapers")


def import_parquet_file(df, doc_type):
    with collection.batch.dynamic() as batch:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Importing {doc_type}"):
            batch.add_object(
                properties={
                    "paperId": row["paperId"],
                    "type": doc_type,
                    "chunkIndex": row.get("chunkIndex", -1),
                    "content": row.get("text", row.get("content", "")),
                    "title": row.get("title", ""),
                },
                vector=row["embedding"].tolist(),
            )


# print(f"Importing abstracts embeddings...")
# meta_df = pd.read_csv(
#     "data/04_data_only_necessary_columns.csv", usecols=["paperId", "title", "abstract"]
# )

# embeddings_df = pd.read_parquet("data/07a_abstracts_embeddings.parquet")
# merged_df = pd.merge(embeddings_df, meta_df, on="paperId", how="inner")
# merged_df["content"] = (
#     "TITLE: " + merged_df["title"] + "\nCONTENT: " + merged_df["abstract"]
# )

# import_parquet_file(merged_df, "ABSTRACT")
# del merged_df
# del embeddings_df
# gc.collect()

# print(f"Importing summary embeddings...")
# summaries_text_list = []
# with open("data/06_summaries_for_embeddings.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         summaries_text_list.append(json.loads(line))
# summaries_text_df = pd.DataFrame(summaries_text_list)

# embeddings_df = pd.read_parquet("data/07b_summaries_embeddings.parquet")
# merged_df = pd.merge(embeddings_df, summaries_text_df, on="paperId", how="inner")
# merged_df = pd.merge(
#     merged_df, meta_df[["paperId", "title"]], on="paperId", how="inner"
# )

# merged_df["content"] = (
#     "TITLE: " + merged_df["title"] + "\nCONTENT: " + merged_df["summary"]
# )


# import_parquet_file(merged_df, "SUMMARY")
# del merged_df
# del embeddings_df
# del summaries_text_df
# del meta_df
# gc.collect()

print("Building lookup from JSONL (title + text)...")
chunk_lookup = {}

with open(f"{CHUNKS_DIR}/05_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        pid = item["paperId"]
        title = item["title"]
        for idx, text in enumerate(item["chunks"]):
            chunk_lookup[(pid, idx)] = (title, text)

print(f"Lookup ready. Mapped {len(chunk_lookup)} chunks.")

chunk_files = [f"{CHUNKS_DIR}/07c_0{i+1}_chunks_embeddings.parquet" for i in range(4)]

print(f"Importing chunks embeddings...")
for f_path in sorted(chunk_files):
    print(f"\n--- Processing {f_path} ---")
    df = pd.read_parquet(f_path)
    print("Enriching data with titles and text...")

    def get_chunk_info(row):
        return chunk_lookup.get((row["paperId"], row["chunkIndex"]), (None, None))

    info_series = df.apply(get_chunk_info, axis=1)

    df["title"] = info_series.apply(lambda x: x[0])
    df["raw_text"] = info_series.apply(lambda x: x[1])

    df["content"] = "TITLE: " + df["title"] + "\nCONTENT: " + df["raw_text"]

    print(f"Importing {len(df)} records to Weaviate...")
    import_parquet_file(df, "CHUNK")

    print(f"Cleaning memory for {f_path}...")
    del df
    del info_series
    gc.collect()

del chunk_lookup
gc.collect()

print("\n[SUCCESS] All chunks are in the database!")


def load_jsonl_to_memory(path):
    """Loads JSONL into a flat lookup dictionary."""
    print(f"Loading {path} into RAM...")
    lookup = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            item = json.loads(line)
            pid = item["paperId"]
            title = item["title"]
            for idx, text in enumerate(item["chunks"]):
                # Using a tuple (title, text) is more RAM-efficient than a dict
                lookup[(pid, idx)] = (title, text)
    return lookup


def run_import_chunks():
    jsonl_path = "data/05_chunks.jsonl"
    small_parts_dir = "data/small_chunks"

    # 1. Load the big lookup
    text_lookup = load_jsonl_to_memory(jsonl_path)
    gc.collect()  # Force GC after heavy loading

    # 2. Process small parquet files
    files = sorted(
        [os.path.join(small_parts_dir, f) for f in os.listdir(small_parts_dir)]
    )

    print(f"\nStarting import of {len(files)} small parts...")

    for f_path in files:
        print(f"Part: {os.path.basename(f_path)}")
        df = pd.read_parquet(f_path)

        with collection.batch.dynamic() as batch:
            for row in df.itertuples(index=False):
                # Get text and title from our lookup
                info = text_lookup.get((row.paperId, row.chunkIndex))
                if info:
                    title, text = info
                    batch.add_object(
                        properties={
                            "paperId": row.paperId,
                            "type": "CHUNK",
                            "chunkIndex": row.chunkIndex,
                            "content": f"TITLE: {title}\nCONTENT: {text}",
                            "title": title,
                        },
                        vector=row.embedding.tolist(),
                    )

        # Clean up memory after each 50k batch
        del df
        gc.collect()

    print("\n[SUCCESS] Import finished!")


# run_import_chunks()
client.close()