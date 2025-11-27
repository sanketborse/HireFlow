import pandas as pd
import chromadb
import uuid
import os


class Portfolio:
    def __init__(self, file_path=None):
        if file_path is None:
            # Default to ../my_portfolio.csv relative to this file
            file_path = os.path.join(os.path.dirname(__file__), "../my_portfolio.csv")

        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)

        # Persistent Chroma client
        self.chroma_client = chromadb.PersistentClient("vectorstore")
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        """
        Load portfolio data into ChromaDB only if not already loaded.
        """
        if self.collection.count() == 0:
            for _, row in self.data.iterrows():
                techstack = str(row["Techstack"])
                links = str(row["Links"])

                self.collection.add(
                    documents=[techstack],
                    metadatas=[{"links": links}],
                    ids=[str(uuid.uuid4())],
                )

    def query_links(self, skills):
        """
        Query ChromaDB using skills as query_texts.
        Handles all weird types (int, None, etc.) safely.
        """

        # Normalize skills into a list of strings
        if skills is None:
            skills = []
        elif isinstance(skills, str):
            skills = [skills]
        elif isinstance(skills, (int, float)):
            skills = []
        elif isinstance(skills, list):
            skills = [str(s) for s in skills if s is not None]
        else:
            skills = []

        if len(skills) == 0:
            return []

        result = self.collection.query(query_texts=skills, n_results=2)
        # result["metadatas"] is usually a list-of-lists
        return result.get("metadatas", [])
