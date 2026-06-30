import os
import sys
from pymongo import MongoClient

class DBStorer:
    def __init__(self, entity_name):
        self.entity = []
        client = MongoClient("mongodb://localhost:27017")
        db = client["plantsim"]
        self.collection = db[entity_name]

    def add_entity(self, entity):
        self.entity.append(entity)

    def save_to_db(self):
        self.collection.delete_many({})
        self.collection.insert_many([entity.to_dict() for entity in self.entity])



