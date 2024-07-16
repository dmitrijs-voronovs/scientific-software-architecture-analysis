from services.MongoDBConnection import MongoDBConnection


def upsert_collection(database: str, collection: str):
    mongo_conn = MongoDBConnection()
    client = mongo_conn.get_client()
    db = client[database]
    collection = db[collection]
    collection.insert_one({"name": "test"})
    print(db.list_collection_names())
    print(collection.find_one({"name": "test"}))
    return collection
