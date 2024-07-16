import os
import time

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from urllib.parse import quote_plus


class MongoDBConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBConnection, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.connect()
        return cls._instance

    def connect(self):
        if self.client is None:
            mongo_host = os.getenv('MONGO_HOST', 'localhost')
            mongo_port = int(os.getenv('MONGO_PORT', 27017))

            self.client = MongoClient(host=mongo_host, port=mongo_port,
                                      username=os.getenv('MONGO_ROOT_USERNAME'),
                                      password=os.getenv('MONGO_ROOT_PASSWORD')
                                      , serverSelectionTimeoutMS=5000)
            # The ismaster command is cheap and does not require auth.
            self.client.admin.command('ismaster')
            print("Successfully connected to MongoDB.")

    def get_client(self) -> MongoClient:
        if self.client is None:
            self.connect()
        return self.client

    def check_database_exists(self, db_name):
        if self.client:
            return db_name in self.client.list_database_names()
        return False

    def close_connection(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")
