import customtkinter as ctk
from login import login
from signup import signup
from home import home

def welcome():
    root = ctk.CTk()

    WIDTH = 400
    HEIGHT = 400
    root.geometry(f"{WIDTH}x{HEIGHT}")

    welcome_label = ctk.CTkLabel(root, text="Welcome to Forecastr!", font=("Arial", 28))
    welcome_label.place(relx=0.5, rely=0.1, anchor=ctk.CENTER)

    def welcome_signup(root, signup, home):
        root.destroy()
        signup(home, welcome)

    def welcome_login(root, login, home):
        root.destroy()
        login(home,welcome)

    signup_button = ctk.CTkButton(root, text="Create an account", command=lambda: welcome_signup(root=root, signup=signup, home=home))
    signup_button.place(relx=0.5, rely=0.2, anchor=ctk.CENTER)

    login_button = ctk.CTkButton(root, text="Login", command=lambda: welcome_login(root=root, login=login, home=home))
    login_button.place(relx=0.5, rely=0.3, anchor=ctk.CENTER)

    root.mainloop()

if __name__ == "__main__":
    welcome()
