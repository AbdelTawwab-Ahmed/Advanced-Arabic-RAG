import os
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion
from src.indexing.qdrant_setup import get_qdrant_client, COLLECTION_NAME
from src.embedding.embedder import embed_dense, embed_sparse

OUTPUT_DIR = "notebooks/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = get_qdrant_client()
total_points = client.count(COLLECTION_NAME).count

query_text = "ما هي إجراءات إصدار البطاقات"

dense_vec = embed_dense([query_text], input_type="search_query")[0]
sparse_vec = embed_sparse([query_text])[0]

results = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=10),
        Prefetch(
            query=SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()),
            using="sparse",
            limit=10,
        ),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=3,
)

output_path = os.path.join(OUTPUT_DIR, "test_retrieval_output.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Total points in collection: {total_points}\n")
    f.write(f"Query: {query_text}\n\n")
    for point in results.points:
        f.write(f"score={point.score:.4f} | {point.payload['source_pdf']} | page {point.payload['page_number']}\n")
        f.write(point.payload['text'][:200])
        f.write("\n\n")

print(f"\nDone — results written to {output_path}")