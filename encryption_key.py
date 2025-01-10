import os
from cryptography.fernet import Fernet

def generate_key():
    if not os.path.exists("secret.key"):
        key = Fernet.generate_key()
        with open("secret.key", "wb") as key_file:
            key_file.write(key)

def load_key():
    return open("secret.key", "rb").read()


