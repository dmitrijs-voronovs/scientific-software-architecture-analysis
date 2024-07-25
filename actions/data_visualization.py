from constants.db import PROJECTS_COLLECTION_NAME, PROJECTS_DB_NAME
from services.MongoDBConnection import MongoDBConnection


def get_schema_sample(collection, sample_size=1000):
    pipeline = [{"$sample": {"size": sample_size}},  # {"$project": {"_id": 0}}  # Exclude _id field
                ]
    sample = list(collection.aggregate(pipeline))
    print(sample)

    schema = {}
    for doc in sample:
        for field, value in doc.items():
            if field not in schema:
                schema[field] = set()
            schema[field].add(type(value).__name__)

    return {field: list(types) for field, types in schema.items()}


def determine_chart_type(field_name, data_types):
    if 'int' in data_types or 'float' in data_types:
        return 'histogram'
    elif 'str' in data_types:
        return 'bar_chart'
    elif 'bool' in data_types:
        return 'pie_chart'
    elif 'datetime' in data_types:
        return 'time_series'
    else:
        return 'table'  # Default to table for complex types


def visualize():
    client = MongoDBConnection().get_client()
    db = client[PROJECTS_DB_NAME]
    # collection = db[PROJECTS_COLLECTION_NAME]
    collection = db['proposal-1']
    # result = client[PROJECTS_DB_NAME][PROJECTS_COLLECTION_NAME].aggregate([{'$sample': {
    #     'size': 1000}}])
    # result = client[PROJECTS_DB_NAME][PROJECTS_COLLECTION_NAME].find({"PRsOpenN": 33})
    result = collection.count_documents({"branchesN": {"$gt": 0}})
    print(result)
    print(db.list_collection_names())
    print(list(collection.find()))
    roles = client.admin.command('usersInfo',
                                 client.admin.command('connectionStatus')['authInfo']['authenticatedUsers'][0]['user'])
    print(roles)

    # schema = get_schema_sample(client['projects']['repos-by-category'])
    # print(schema)
    # chart_types = {field: determine_chart_type(field, types) for field, types in schema.items()}
    # print(chart_types)
