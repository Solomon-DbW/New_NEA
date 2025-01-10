import customtkinter as ctk
from password_encryption import decrypt_password
from database_manager import session, User

# Login Page
def login(home, welcome):

    # GUI Setup
    root = ctk.CTk()
    WIDTH = 400
    HEIGHT = 400
    root.geometry(f"{WIDTH}x{HEIGHT}")

    # Entry Box For Username
    username_entry = ctk.CTkEntry(root, placeholder_text="Enter your username: ")
    username_entry.place(relx=0.5, rely=0.1, anchor=ctk.CENTER)
    
    # Enrty Box For Password
    password_entry = ctk.CTkEntry(root, placeholder_text="Enter your password: ", show="*")
    password_entry.place(relx=0.5, rely=0.2, anchor=ctk.CENTER)

    # Show Password Option Functionality
    show_password = ctk.BooleanVar()
    show_password_check = ctk.CTkCheckBox(root, text="Show password", variable=show_password, command=lambda: password_entry.configure(show="" if show_password.get() else "*"))
    show_password_check.place(relx=0.5, rely=0.3, anchor=ctk.CENTER)

    # Function To Submit Signup Details
    def submit_login():

        # Gets Text From Entry Boxes
        username = username_entry.get()
        password = password_entry.get()

        # Searches Database For User With Same Username As Inputted Username
        user = session.query(User).filter_by(username=username).first()
        if user:
            try:
                stored_password = decrypt_password(user.password)  # Decrypt stored password
                if stored_password == password:
                    ctk.CTkLabel(root, text="Login successful!").place(relx=0.5, rely=0.4, anchor=ctk.CENTER)
                    current_user_id = user.userid # Stores the current user's id
                    current_username = user.username # Stores the current username

                    # Storing User Credentials in Text File
                    with open("user_id.txt", "w") as f:
                        f.write(str(current_user_id))
                        f.write(f"\n{str(current_username)}")

                    root.destroy()
                    home(current_user_id)  # Call home window after login
                else:
                    ctk.CTkLabel(root, text="Incorrect password").place(relx=0.5, rely=0.4, anchor=ctk.CENTER)
            except Exception as e: # Error Handling Functionality
                ctk.CTkLabel(root, text=f"Decryption error: {e}").place(relx=0.5, rely=0.4, anchor=ctk.CENTER)
        else:
            ctk.CTkLabel(root, text="Username not found").place(relx=0.5, rely=0.4, anchor=ctk.CENTER)

    # Function To Allow User To Return To Welcome Page
    def return_to_welcome():
        root.destroy()
        welcome()

    # Login Page Buttons Configuration
    back_button = ctk.CTkButton(root, text="Back", command=lambda:return_to_welcome())
    back_button.place(relx=0.5, rely=0.4, anchor=ctk.CENTER)

    login_button = ctk.CTkButton(root, text="Log in", command=submit_login)
    login_button.place(relx=0.5, rely=0.35, anchor=ctk.CENTER)
    
    root.mainloop()


