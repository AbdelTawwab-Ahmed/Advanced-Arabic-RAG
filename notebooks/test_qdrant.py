from src.indexing.qdrant_setup import get_qdrant_client, create_collection


client = get_qdrant_client()

collection = create_collection(client)

print("\n", client.get_collections())