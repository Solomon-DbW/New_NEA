import customtkinter as ctk
from login import login
from signup import signup
from home import home

def welcome():
    root = ctk.CTk() # Create a new window

    WIDTH = 400
    HEIGHT = 400
    root.geometry(f"{WIDTH}x{HEIGHT}") # Set the size of the window

    welcome_label = ctk.CTkLabel(root, text="Welcome to Forecastr!", font=("Arial", 28)) # Create a label to welcome the user
    welcome_label.place(relx=0.5, rely=0.1, anchor=ctk.CENTER)

    def welcome_signup(root, signup, home): # Function to navigate to the signup screen
        root.destroy()
        signup(home, welcome)

    def welcome_login(root, login, home): # Function to navigate to the login screen
        root.destroy()
        login(home,welcome)

    signup_button = ctk.CTkButton(root, text="Create an account", command=lambda: welcome_signup(root=root, signup=signup, home=home)) # Button to navigate to the signup screen
    signup_button.place(relx=0.5, rely=0.2, anchor=ctk.CENTER)

    login_button = ctk.CTkButton(root, text="Login", command=lambda: welcome_login(root=root, login=login, home=home)) # Button to navigate to the login screen
    login_button.place(relx=0.5, rely=0.3, anchor=ctk.CENTER)

    root.mainloop()

if __name__ == "__main__":
    welcome() # Call the welcome function
