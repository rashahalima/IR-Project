import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import processing_quare as qp

class SearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Search")
        
        self.dataset_label = ttk.Label(root, text="Choose Dataset:")
        self.dataset_label.pack(pady=5)

        self.dataset_combobox = ttk.Combobox(root, values=["science", "antique"])
        self.dataset_combobox.pack(pady=5)

        self.query_label = ttk.Label(root, text="Enter Query:")
        self.query_label.pack(pady=5)

        self.query_entry = ttk.Entry(root, width=50)
        self.query_entry.pack(pady=5)

        self.search_button = ttk.Button(root, text="Search", command=self.search)
        self.search_button.pack(pady=10)

        self.results_text = tk.Text(root, width=80, height=20)
        self.results_text.pack(pady=10)

        self.datasets = {
            "science": {
                "processed_data_path": "C:/Users/Evo.Store/Desktop/lotte/science/dev/collection1.tsv",
                "acronym_file_path": "C:/Users/Evo.Store/Desktop/lotte/science/dev/acronyms"
            },
            "antique": {
                "processed_data_path": "C:/Users/Evo.Store/Desktop/antique/collection1.tsv",
                "acronym_file_path": "C:/Users/Evo.Store/Desktop/antique/acronyms"
            }
        }

    def read_processed_data(self, file_path):
        df = pd.read_csv(file_path, sep='\t', header=None)
        df.columns = ['Document Number', 'Content']
        df['Content'] = df['Content'].fillna('')
        return df

    def retrieve_documents(self, query, vectorizer, document_vectors, document_ids, acronym_file_path):
        acronym_dict = qp.create_acronym_dic(acronym_file_path)
        query_vector = vectorizer.transform([qp.process_text(query, acronym_dict)])
        cos_similarities = cosine_similarity(query_vector, document_vectors).flatten()
        ranked_document_indices = np.argsort(cos_similarities)[::-1]
        retrieved_docs = [document_ids[idx] for idx in ranked_document_indices[:10]]
        return retrieved_docs

    def search(self):
        dataset_name = self.dataset_combobox.get()
        query = self.query_entry.get()

        if not dataset_name or not query:
            messagebox.showwarning("Input Error", "Please select a dataset and enter a query.")
            return

        dataset = self.datasets.get(dataset_name)
        if not dataset:
            messagebox.showwarning("Dataset Error", "Invalid dataset selected.")
            return

        processed_data_path = dataset["processed_data_path"]
        acronym_file_path = dataset["acronym_file_path"]

        processed_data = self.read_processed_data(processed_data_path)
        vectorizer = TfidfVectorizer(stop_words='english')
        document_vectors = vectorizer.fit_transform(processed_data['Content'])
        document_ids = processed_data['Document Number'].tolist()

        retrieved_docs = self.retrieve_documents(query, vectorizer, document_vectors, document_ids, acronym_file_path)

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Retrieved Documents:\n")
        self.results_text.insert(tk.END, "\n".join(retrieved_docs))

if __name__ == "__main__":
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()
