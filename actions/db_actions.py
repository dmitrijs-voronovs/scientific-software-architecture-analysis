from services.MongoDBConnection import MongoDBConnection


def upsert_collection(database: str, collection: str):
    mongo_conn = MongoDBConnection()
    client = mongo_conn.get_client()
    db = client[database]
    collection = db[collection]
    # collection.insert_one({"name": "test"})
    collection.update_one({"_id": "111"}, {"$set": {"name": "test2"}}, upsert=True)
    # collection.drop()
    print(db.list_collection_names())
    print(list(collection.find()))
    return collection


async def upsert_collection_asnyc(database: str, collection: str):
    mongo_conn = MongoDBConnection()
    client = mongo_conn.get_client()
    db = client[database]
    collection = db[collection]
    # collection.insert_one({"name": "test"})
    collection.update_one({"_id": "111"}, {"$set": {"name": "test2"}}, upsert=True)
    # collection.drop()
    print(db.list_collection_names())
    print(list(collection.find()))
    return collection
