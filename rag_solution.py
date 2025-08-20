import numpy as np
import datasets
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import faiss
import json

# Split filing into overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   
    chunk_overlap=50
)

# embed text using OpenAI pretrained model
def embed_texts(texts, model="text-embedding-3-small"):
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype="float32")

# Save embeddings in vector database
def index_embeddings(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

## End of retriever, start of extractor

# Search index for most similar chunks to query
def search_index(query, index, chunks, n):
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, k=n)
    top_matches = [chunks[i] for i in indices[0]]
    return distances, indices, top_matches

# Extract variable from context chunks
def extract_variable(query, doc, contexts):
    client = OpenAI()
    context = "\n".join(contexts)
    messages = [
        {"role": "system", "content": "Extract the requested fields as JSON. If unknown, use null. Only use data from the context provided. Respond with one field only. Don't include year in variable name."},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content) 

## Put it all together to one call

def main():
    filings = datasets.load_from_disk("aig_filings")
    results = []
    for filing in filings:
        doc =  "\n\n".join([filing.get(section) for section in filings.column_names[3:]])

        chunks = splitter.split_text(doc)
        embeds = embed_texts(chunks[0:2000]) #limit for OpenAI - change to multiple calls
        idxs = index_embeddings(embeds)

        year =  filing.get("year")
        queries = [
            "What industry does AIG operate in?",
            "How many employees did AIG have in year " + str(year) + "?",
            "What are the total assets of AIG in year " + str(year) + "?"
        ]

        filing_dict = {"year": year}
        for query in queries:
            dist, idx, contxts = search_index(query, idxs, chunks, 3)
            print(contxts)
            var = extract_variable(query, doc, contxts)
            filing_dict[list(var.keys())[0]] = var[list(var.keys())[0]]
            
        results.append(filing_dict)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()