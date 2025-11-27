import os
import uuid
import pandas as pd
import chromadb

os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"


class Portfolio:
    def __init__(self, file_path=None):
        if file_path is None:
            # 🔥 Look for my_portfolio.csv in the SAME folder as this file
            file_path = os.path.join(os.path.dirname(__file__), "my_portfolio.csv")

        self.file_path = file_path

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Portfolio CSV not found at: {self.file_path}")

        self.data = pd.read_csv(self.file_path)

        self.chroma_client = chromadb.PersistentClient("vectorstore")
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    # rest of your methods stay same...
