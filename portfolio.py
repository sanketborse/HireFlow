import os
import uuid
import pandas as pd
import chromadb

# 🔇 Disable Chroma telemetry noise
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"


class Portfolio:
    def __init__(self, file_path=None):
        if file_path is None:
            # Default my_portfolio.csv one level up from this file
            file_path = os.path.join(os.path.dirname(__file__), "../my_portfolio.csv")

        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)

        self.chroma_client = chromadb.PersistentClient("vectorstore")
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        """
        Load portfolio data into ChromaDB only once.
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
        Always normalizes skills so Chroma never receives an int.
        """

        # --- DEBUG (optional, you can keep for now) ---
        print("DEBUG skills BEFORE normalize:", repr(skills), type(skills))

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

        print("DEBUG skills AFTER normalize:", repr(skills), type(skills))

        if len(skills) == 0:
            return []

        result = self.collection.query(query_texts=skills, n_results=2)
        return result.get("metadatas", [])
