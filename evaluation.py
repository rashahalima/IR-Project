import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import processing_quare as qp

def read_processed_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns = ['Document Number', 'Content']
    df['Content'] = df['Content'].fillna('')
    return df

def read_queries(file_path):
    queries = {}
    with open(file_path) as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                query_id, query_text = parts
                queries[query_id] = query_text
            else:
                print(f"Skipping invalid line: {line.strip()}")
    return queries


def evaluate(queries, relevant_docs, retrieved_docs):
    precision_scores = []
    recall_scores = []
    average_precision_scores = []
    precision_at_10_scores = []
    reciprocal_rank_scores = []

    for query_id, relevant_doc_ids in relevant_docs.items():
        relevant_set = set(relevant_doc_ids)
        retrieved_list = retrieved_docs.get(query_id, [])

        if not relevant_set:
            print(f"No relevant documents for query {query_id}")
            continue

        if not retrieved_list:
            print(f"No retrieved documents for query {query_id}")
            continue

        true_positives = sum(1 for doc_id in retrieved_list if doc_id in relevant_set)
        recall = true_positives / len(relevant_set)
        print(f"Recall for query {query_id}: {recall}")
        recall_scores.append(recall)

        y_true = [1 if doc_id in relevant_set else 0 for doc_id in retrieved_list]
        y_pred = [1] * len(retrieved_list)

        if sum(y_true) == 0:
            print(f"No relevant documents in retrieved set for query {query_id}")
            continue

        precision = precision_score(y_true, y_pred, zero_division=0)
        print(f"Precision for query {query_id}: {precision}")

        avg_precision = 0
        relevant_retrieved = 0
        for i, doc_id in enumerate(retrieved_list):
            if doc_id in relevant_set:
                relevant_retrieved += 1
                avg_precision += relevant_retrieved / (i + 1)
        avg_precision /= len(relevant_set)

        y_true_10 = y_true[:10]
        precision_at_10 = sum(y_true_10) / 10  
        if any(doc in retrieved_list for doc in relevant_set):
            reciprocal_rank = 1 / (retrieved_list.index(next(doc for doc in relevant_set if doc in retrieved_list)) + 1)
        else:
            reciprocal_rank = 0

        precision_scores.append(precision)
        average_precision_scores.append(avg_precision)
        precision_at_10_scores.append(precision_at_10)
        reciprocal_rank_scores.append(reciprocal_rank)

    mean_recall = np.mean(recall_scores)
    mean_precision = np.mean(precision_scores)
    mean_average_precision = np.mean(average_precision_scores)
    mean_precision_at_10 = np.mean(precision_at_10_scores)
    mean_reciprocal_rank = np.mean(reciprocal_rank_scores)

    return mean_recall, mean_precision, mean_average_precision, mean_precision_at_10, mean_reciprocal_rank

def retrieve_documents(query, vectorizer, document_vectors, document_ids, acronym_file_path):
    acronym_dict = qp.create_acronym_dic(acronym_file_path)
    query_vector = vectorizer.transform([qp.process_text(query, acronym_dict)])
    cos_similarities = cosine_similarity(query_vector, document_vectors).flatten()
    ranked_document_indices = np.argsort(cos_similarities)[::-1]

    retrieved_docs = [document_ids[idx] for idx in ranked_document_indices[:10]]
    return retrieved_docs

def process_dataset(processed_data_path, queries_path, acronym_file_path, qrel_file_path):
    processed_data = read_processed_data(processed_data_path)
    queries = read_queries(queries_path)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    document_vectors = vectorizer.fit_transform(processed_data['Content'])
    document_ids = processed_data['Document Number'].tolist()

    retrieved_documents = {}
    for query_id, query_text in queries.items():
        retrieved_docs = retrieve_documents(query_text, vectorizer, document_vectors, document_ids, acronym_file_path)
        retrieved_documents[query_id] = retrieved_docs

    relevant_docs = {}
    with open(qrel_file_path, "r") as file:
        for line in file:
            record = json.loads(line)
            relevant_docs[str(record['qid'])] = record['answer_pids']

    mean_recall, mean_precision, mean_average_precision, mean_precision_at_10, mean_reciprocal_rank = evaluate(queries, relevant_docs, retrieved_documents)

    return mean_recall, mean_precision, mean_average_precision, mean_precision_at_10, mean_reciprocal_rank, queries, relevant_docs, retrieved_documents

def main():
    datasets = [
        {
            "name": "science",
            "processed_data_path": "C:/Users/Evo.Store/Desktop/lotte/science/dev/collection1.tsv",
            "queries_path": "C:/Users/Evo.Store/Desktop/lotte/science/dev/questions.search1.tsv",
            "acronym_file_path": "C:/Users/Evo.Store/Desktop/lotte/science/dev/acronyms",
            "qrel_file_path": "C:/Users/Evo.Store/Desktop/lotte/science/dev/qas.search.jsonl"
        },
        {
            "name": "antique",
            "processed_data_path": "C:/Users/Evo.Store/Desktop/antique/collection1.tsv",
            "queries_path": "C:/Users/Evo.Store/Desktop/antique/questions.search1.tsv",
            "acronym_file_path": "C:/Users/Evo.Store/Desktop/antique/acronyms",
            "qrel_file_path": "C:/Users/Evo.Store/Desktop/antique/output.jsonl"
        }
    ]

    for dataset in datasets:
        print(f"Processing dataset: {dataset['name']}")
        mean_recall, mean_precision, mean_average_precision, mean_precision_at_10, mean_reciprocal_rank, queries, relevant_docs, retrieved_documents = process_dataset(
            dataset["processed_data_path"],
            dataset["queries_path"],
            dataset["acronym_file_path"],
            dataset["qrel_file_path"]
        )

        print(f"Results for dataset: {dataset['name']}")
        print("Mean Precision:", mean_precision)
        print("Mean Recall:", mean_recall)
        print("Mean Average Precision:", mean_average_precision)
        print("Mean Precision@10:", mean_precision_at_10)
        print("Mean Reciprocal Rank:", mean_reciprocal_rank)
        print()

        # for query_id in queries.keys():
        #     print(f"Query ID: {query_id}")
        #     print(f"Relevant Documents: {relevant_docs.get(query_id, [])}")
        #     print(f"Retrieved Documents: {retrieved_documents.get(query_id, [])}")
        #     print()

if __name__ == "__main__":
    main()
