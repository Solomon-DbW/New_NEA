import customtkinter as ctk
from password_encryption import encrypt_password
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_manager import User, Base

def signup(home,welcome):

    # GUI Setup
    root = ctk.CTk()
    WIDTH = 400
    HEIGHT = 400
    root.geometry(f"{WIDTH}x{HEIGHT}")

    # Entry Box For Username
    username_entry = ctk.CTkEntry(root, placeholder_text="Create your username ")
    username_entry.place(relx=0.5, rely=0.1, anchor=ctk.CENTER)

    # Entry Box For Password
    password_entry = ctk.CTkEntry(root, placeholder_text="Create your password ", show="*")
    password_entry.place(relx=0.5, rely=0.2, anchor=ctk.CENTER)

    # Show Password Option Functionality
    show_password = ctk.BooleanVar()
    show_password_check = ctk.CTkCheckBox(
        root, 
        text="Show password", 
        variable=show_password, 
        command=lambda: password_entry.configure(show="" if show_password.get() else "*")
    )
    show_password_check.place(relx=0.5, rely=0.3, anchor=ctk.CENTER)

    # Function To Submit Signup Details
    def submit_signup():

        # Gets Text From Entry Boxes
        username = username_entry.get()
        password = encrypt_password(password_entry.get())  # Encrypt password

        # Checks If Username Or Password Is Empty When User Clicks Submit
        while len(username) == 0 or len(password_entry.get()) == 0:
            ctk.CTkLabel(root, text="Please fill in all fields.").place(relx=0.5, rely=0.4, anchor=ctk.CENTER)
            return

        # Establishes Database Connection
        engine = create_engine("sqlite:///users_and_details.db")
        Base.metadata.create_all(bind=engine)

        Session = sessionmaker(bind=engine)
        session = Session()

        # Checks If Username Is Already Taken In The Database
        existing_user = session.query(User).filter_by(username=username).first()
        if existing_user:
            ctk.CTkLabel(root, text="Username already taken.").place(relx=0.5, rely=0.4, anchor=ctk.CENTER)
            return
        else:
            # Saves User Credentials In Database
            user = User(username, password)
            session.add(user)
            session.commit()
            ctk.CTkLabel(root, text="Signup successful!").place(relx=0.5, rely=0.4, anchor=ctk.CENTER)
        
            current_user_id = user.userid # Gets The Current User's ID

            # Storing User Credentials in Text File
            with open("user_id.txt", "w") as f:
                f.write(str(current_user_id))
                f.write(f"\n{str(username)}")


        # Closes Database Connection
        session.close()
        home(username) # Calls The Homepage

    # Function To Allow User To Return To Welcome Page
    def return_to_welcome():
        root.destroy()
        welcome()

    # Signup Page Buttons Configuration
    signup_button = ctk.CTkButton(root, text="Sign up", command=submit_signup)
    signup_button.place(relx=0.5, rely=0.35, anchor=ctk.CENTER)

    return_to_welcome_button = ctk.CTkButton(root, text="Return to welcome", command=lambda: return_to_welcome())
    return_to_welcome_button.place(relx=0.5, rely=0.4, anchor=ctk.CENTER)
    
    root.mainloop()
