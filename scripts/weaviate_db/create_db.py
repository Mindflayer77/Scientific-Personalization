import weaviate
import weaviate.classes as wvc

client = weaviate.connect_to_local()

def create_compressed_collection():
    if client.collections.exists("ResearchPapers"):
        client.collections.delete("ResearchPapers")

    client.collections.create(
        name="ResearchPapers",
        vectorizer_config=None,
        properties=[
            wvc.config.Property(name="paperId", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="chunkIndex", data_type=wvc.config.DataType.INT),
        ],
        vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE,
            quantizer=wvc.config.Configure.VectorIndex.Quantizer.pq(
                segments=512, centroids=256, training_limit=100000
            ),
        ),
    )
    print("Compressed collection created.")


create_compressed_collection()
client.close()