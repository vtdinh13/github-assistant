from typing import List, Any
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')



class SearchTool:
    def __init__(self, index):
        self.index=index

    def search(self, query: str) -> List[Any]:
        """
        Perform a text-based search on the Elastic Search index.

        Args:
            query (str): The search query string.

        Returns:
            List[Any]: A list of up to 5 search results returned by the Elastic Search index.
        """
        query_embedding = embedding_model.encode(query)
        return self.index.search(query_embedding, num_results=5)