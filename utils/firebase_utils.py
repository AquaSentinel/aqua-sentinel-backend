# firebase_utils.py
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Load .env
load_dotenv()

cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Initialize Firebase app only once
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        "projectId": os.getenv("FIREBASE_PROJECT_ID"),
        "storageBucket": f"{os.getenv('FIREBASE_PROJECT_ID')}.firebasestorage.app"
    })

# Firestore & Storage clients
db = firestore.client()
bucket = storage.bucket()

print("Firebase initialized successfully.")

