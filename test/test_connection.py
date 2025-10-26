# test/test_connection.py
from utils.firebase_utils import db

users = db.collection("users").limit(1).stream()
for u in users:
    print(u.id, u.to_dict())
print("Firestore connection successful!")
