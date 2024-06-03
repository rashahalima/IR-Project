import json

# مسارات الملفات
qrels_file_path = 'C:/Users/Evo.Store/Desktop/antique/train/qrels'
queries_file_path = 'C:/Users/Evo.Store/Desktop/antique/train/queries.txt'
output_file_path = 'C:/Users/Evo.Store/Desktop/antique/output.jsonl'

# قراءة ملف queries.txt وتحويله إلى قاموس
queries = {}
with open(queries_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        qid = int(parts[0])
        query = parts[1]
        queries[qid] = query

# قراءة ملف qrels.txt وتحويله إلى صيغة مناسبة
qrels = {}
with open(qrels_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        qid = int(parts[0])
        doc_id = parts[2]
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(doc_id)

# كتابة البيانات إلى ملف JSONL
with open(output_file_path, 'w') as outfile:
    for qid, query in queries.items():
        if qid in qrels:
            json_object = {
                "qid": qid,
                "query": query,
                "url": "",  # يمكن إضافة URL هنا إذا كان متاحاً
                "answer_pids": qrels[qid]
            }
            outfile.write(json.dumps(json_object, ensure_ascii=False) + '\n')

print(f"تم تحويل البيانات وحفظها في الملف {output_file_path}")