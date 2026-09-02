from src.embedding.embedder import embed_dense, embed_sparse


text = ["مرحبا بكم في بنك الإسكان", "هذا اختبار للتضمين"]

dense_vectors = embed_dense(text)
print(f"\n\nGot {len(dense_vectors)} vectors, each of dimension {len(dense_vectors[0])}")


sparse_vectors = embed_sparse(text)
for v in sparse_vectors:
    print(f"\n\nNon-zero terms: {len(v.indices)}")