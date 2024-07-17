import asyncio

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


async def upsert_collection_async(database: str, collection: str, data):
    try:
        mongo_conn = MongoDBConnection()
        client = mongo_conn.get_client()
        db = client[database]
        c = db[collection]
        # c.insert_many(data)
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: c.insert_many(data)
        )
    except Exception as e:
        print(e)
        return None
