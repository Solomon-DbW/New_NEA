from cryptography.fernet import Fernet
from encryption_key import load_key

key = load_key() # Load the encryption key
cipher_suite = Fernet(key) # Create a Fernet cipher suite

def encrypt_password(password): # Encrypt the password
    return cipher_suite.encrypt(password.encode()).decode() # Encode the password and return the encrypted password

def decrypt_password(encrypted_password): # Decrypt the password
    return cipher_suite.decrypt(encrypted_password.encode()).decode() # Decode the encrypted password and return the decrypted password

