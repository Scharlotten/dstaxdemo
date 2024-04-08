from astrapy.db import AstraDB, AstraDBCollection
from dotenv import load_dotenv
import os
from models import Product
import json
import ibis
import numpy as np
import pandas as pd
import sqlalchemy as sa
import torch
from datasets import load_dataset
from pandarallel import pandarallel
from transformers import AutoTokenizer
from transformers import AutoModel


def create_connection():
    load_dotenv()
    # Initialize the client
    db = AstraDB(
        token=os.getenv("ASTRA_DB_APP_TOKEN"),
        api_endpoint="https://956bddef-61f8-41c8-8e41-3609f71ccb1d-us-east-1.apps.astra.datastax.com",
        namespace="scraper_app"
    )
    if "product" in db.get_collections().get('status').get('collections'):
        print("Found collection")
    else:
        db.create_collection(collection_name="product",
                                   dimension=5)

    collection = AstraDBCollection(
        collection_name="product", astra_db=db
    )

    print(f"Connected to Astra DB: {db.get_collections()}")
    return db, collection


# collection.insert_one(
#     {"asin": "123AWS", "title":"This is a title"}
# )
# collection.find_one({"asin": "123AWS"})