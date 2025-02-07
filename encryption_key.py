import os
from cryptography.fernet import Fernet

def generate_key(): # Generate a key and save it to a file
    if not os.path.exists("secret.key"):
        key = Fernet.generate_key()
        with open("secret.key", "wb") as key_file:
            key_file.write(key)

def load_key(): # Load the previously generated key
    return open("secret.key", "rb").read()


