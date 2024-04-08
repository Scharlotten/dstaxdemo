from astrapy.db import AstraDB, AstraDBCollection
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import AutoModel
import ast
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def create_connection(coll_name:str):
    load_dotenv()
    # Initialize the client
    db = AstraDB(
        token=os.getenv("ASTRA_DB_APP_TOKEN"),
        api_endpoint="https://956bddef-61f8-41c8-8e41-3609f71ccb1d-us-east-1.apps.astra.datastax.com",
        namespace="scraper_app"
    )
    if coll_name in db.get_collections().get('status').get('collections'):
        print("Found collection continuing")
        res0 = db.delete_collection(collection_name=coll_name)
        db.create_collection(collection_name=coll_name,dimension=384)
    else:
        db.create_collection(collection_name=coll_name,
                                   dimension=384)

    collection = AstraDBCollection(
        collection_name=coll_name, astra_db=db
    )

    print(f"Connected to Astra DB: {db.get_collections()}")
    return db, collection

def get_embedding(sentence: str) -> np.ndarray[np.float32]:
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
    return np.array(embedding, dtype='<f4')


def create_sample_data():
    return pd.DataFrame({"word": ["finance", "bankruptcy", "fraud", "credit",
                                  "delay", "lawyer", "commission", "shady",
                                  "mortgage", "banker", "human", "coffee",
                                  "latte", "milk", "queen", "prince",
                                  "princess", "girl", "monarch",
                                  "constitution"]})
def show_words_corr(df: pd.DataFrame):
    matrix = df["$vector"]
    matrix = matrix.tolist()
    matrix = np.asarray(matrix)


    # Create a t-SNE model and transform the data
    tsne = TSNE(n_components=2, perplexity=4, init='pca',
                learning_rate=400)
    vis_dims = tsne.fit_transform(matrix)

    labels = []
    x = [x for x, y in vis_dims]
    y = [y for x, y in vis_dims]
    for word in range(len(df)):
        first_value = df['word'].iat[word]
        labels.append(first_value)
    print(len(df))
    plt.scatter(x, y, alpha=0.3)
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


if __name__ == "__main__":
    db, collection = create_connection("finance")
    print("Successfully created DB and connection")

    df = create_sample_data()
    print("Successfully created dataframe")

    df['$vector'] = df['word'].apply(get_embedding)
    print("Successfully applied function to generate word embeddings")
    print(df)

    documents = df.to_json(orient="records")
    documents = ast.literal_eval(documents)


    res = collection.insert_many(documents)
    print(res)
    print("successfully inserted records to AstraDB")
    testword = "cheating"
    vector = get_embedding(testword)
    v = vector.tolist()
    results = collection.vector_find(v, limit=2, fields=["word", "$vector"])
    closest_word = results[0].get("word")
    second_closest = results[1].get("word")
    print(f"Similar words are {closest_word} and {second_closest}")
    show_words_corr(df)



