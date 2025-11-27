import os
import uuid
import pandas as pd
import chromadb

# Optional: turn off Chroma telemetry noise in logs
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"


class Portfolio:
    def __init__(self, file_path=None):
        # By default, look for my_portfolio.csv in the SAME folder as this file
        if file_path is None:
            file_path = os.path.join(os.path.dirname(__file__), "my_portfolio.csv")

        self.file_path = file_path

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Portfolio CSV not found at: {self.file_path}")

        # Load CSV
        self.data = pd.read_csv(self.file_path)

        # ChromaDB setup
        self.chroma_client = chromadb.PersistentClient("vectorstore")
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        """
        Insert portfolio rows into Chroma only once.
        """
        if self.collection.count() == 0:
            documents = []
            metadatas = []
            ids = []

            for _, row in self.data.iterrows():
                techstack = str(row["Techstack"])
                links = str(row["Links"])

                documents.append(techstack)
                metadatas.append({"links": links})
                ids.append(str(uuid.uuid4()))

            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

    def query_links(self, skills):
        """
        Query Chroma using skills. Make sure skills is ALWAYS a list of strings,
        so we never hit 'object of type int has no len()'.
        """

        # Debug (shows in Render logs)
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
