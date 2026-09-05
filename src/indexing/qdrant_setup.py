from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from src import config


COLLECTION_NAME = config.QDRANT_COLLECTION_NAME
DENSE_VECTOR_SIZE = config.DENSE_VECTOR_SIZE 


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )


def create_collection(client: QdrantClient, recreate: bool = False):
    if recreate and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        print(f"\nCreated collection '{COLLECTION_NAME}'.")
    else:
        print(f"\nCollection '{COLLECTION_NAME}' already exists.")