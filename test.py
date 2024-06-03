import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# مسارات الملفات
output_file_path1 = "C:/Users/Evo.Store/Desktop/lotte/science/dev/questions.search1.tsv"
vector_value_quary_file_path = "C:/Users/Evo.Store/Desktop/lotte/science/dev/vector_value_quary.tsv"
vector_word_quary_file_path = "C:/Users/Evo.Store/Desktop/lotte/science/dev/vector_word_quary.tsv"
output_file_path = "C:/Users/Evo.Store/Desktop/lotte/science/dev/collection1.tsv"
vector_value_file_path = "C:/Users/Evo.Store/Desktop/lotte/science/dev/vector_value.tsv"
vector_word_file_path = "C:/Users/Evo.Store/Desktop/lotte/science/dev/vector_word.tsv"
# print("////////////////////////data set antique/////////////////////")
output_file_path1_antique ="C:/Users/Evo.Store/Desktop/antique/questions.search1.tsv"
vector_value_quary_file_path_antique = "C:/Users/Evo.Store/Desktop/antique/vector_value_quary.tsv"
vector_word_quary_file_path_antique = "C:/Users/Evo.Store/Desktop/antique/vector_word_quary.tsv"
output_file_path_antique = "C:/Users/Evo.Store/Desktop/antique/collection1.tsv"
vector_value_file_path_antique = "C:/Users/Evo.Store/Desktop/antique/vector_value.tsv"
vector_word_file_path_antique = "C:/Users/Evo.Store/Desktop/antique/vector_word.tsv"

def save_sparse_matrix_to_file(sparse_matrix, file_path):
    from scipy.io import mmwrite
    mmwrite(file_path, sparse_matrix)

def save_feature_names_to_file(feature_names, file_path):
    df_words = pd.DataFrame(feature_names)
    df_words.to_csv(file_path, sep='\t', index=False, header=False)

def build_vectors(query_file_path, document_file_path, vector_value_query_path, vector_word_query_path, vector_value_doc_path, vector_word_doc_path):
    query_df = pd.read_csv(query_file_path, sep='\t', header=None)
    query_df.columns = ['Query ID', 'Content']
    document_df = pd.read_csv(document_file_path, sep='\t', header=None)
    document_df.columns = ['Document ID', 'Content']

    query_df = query_df.fillna({'Content': ''})
    document_df = document_df.fillna({'Content': ''})

    tfidf_vectorizer = TfidfVectorizer()

    combined_texts = pd.concat([query_df['Content'], document_df['Content']])

    combined_vectors = tfidf_vectorizer.fit_transform(combined_texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    query_vectors = combined_vectors[:len(query_df)]
    document_vectors = combined_vectors[len(query_df):]

    save_sparse_matrix_to_file(query_vectors, vector_value_query_path)
    save_sparse_matrix_to_file(document_vectors, vector_value_doc_path)
    save_feature_names_to_file(feature_names, vector_word_query_path)
    save_feature_names_to_file(feature_names, vector_word_doc_path)

    return query_vectors, document_vectors, query_df['Query ID'].values, document_df['Document ID'].values

query_vectors, document_vectors, query_ids, document_ids = build_vectors(
    output_file_path1, output_file_path, vector_value_quary_file_path, vector_word_quary_file_path, vector_value_file_path, vector_word_file_path
)

def match_and_rank(query_vectors, document_vectors, query_index):
    similarities = cosine_similarity(query_vectors[query_index], document_vectors)
    ranked_documents_indices = np.argsort(similarities, axis=1)[:, ::-1]
    relevant_documents = ranked_documents_indices[0][:10]
    return relevant_documents

retrieved_documents_list = []
num_queries = query_vectors.shape[0]
for query_index in range(num_queries):
    retrieved_documents_indices = match_and_rank(query_vectors, document_vectors, query_index)
    retrieved_documents = document_ids[retrieved_documents_indices]
    retrieved_documents_list.append(retrieved_documents)
    print(f"Query {query_ids[query_index]} Retrieved Documents: {retrieved_documents}")

output_file_path = "C:/Users/Evo.Store/Desktop/lotte/science/dev/retrieved_documents.tsv"
with open(output_file_path, 'w') as file:
    for query_id, retrieved_docs in zip(query_ids, retrieved_documents_list):
        file.write(f"Query {query_id} Retrieved Documents: {', '.join(map(str, retrieved_docs))}\n")

print("////////////////////////data set antique/////////////////////")

query_vectors, document_vectors, query_ids, document_ids = build_vectors(
    output_file_path1_antique, output_file_path_antique, vector_value_quary_file_path_antique, vector_word_quary_file_path_antique, vector_value_file_path_antique, vector_word_file_path_antique
)

retrieved_documents_list = []
num_queries = query_vectors.shape[0]
for query_index in range(num_queries):
    retrieved_documents_indices = match_and_rank(query_vectors, document_vectors, query_index)
    retrieved_documents = document_ids[retrieved_documents_indices]
    retrieved_documents_list.append(retrieved_documents)
    print(f"Query {query_ids[query_index]} Retrieved Documents: {retrieved_documents}")

output_file_path = "C:/Users/Evo.Store/Desktop/antique/retrieved_documents.tsv"
with open(output_file_path, 'w') as file:
    for query_id, retrieved_docs in zip(query_ids, retrieved_documents_list):
        file.write(f"Query {query_id} Retrieved Documents: {', '.join(map(str, retrieved_docs))}\n")
