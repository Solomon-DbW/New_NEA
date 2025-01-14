import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import re
from datetime import datetime
from database_manager import Card, User, session
from sqlalchemy.exc import SQLAlchemyError


class BankAccountManager:
    def __init__(self, home, homeroot, current_username):
        self.root = ctk.CTk()
        self.homeroot = homeroot
        homeroot.withdraw()
        self.home = home
        self.current_username = current_username  # Store the current_username
        self.root.geometry("800x600")
        self.root.title("Manage Bank Cards")
        self.setup_gui()

    def return_home(self):
        try:
            with open("user_id.txt", "r") as f:
                current_user_id = int(f.readline().strip())
                
            user = User.get_user_by_id(current_user_id)
            if user:
                # self.home(current_user_id)
                self.homeroot.deiconify()
                self.root.destroy()
            else:
                print("Error: User not found.")
        except Exception as e:
            print(f"Error in return_home: {e}")


    def setup_gui(self):
        self.notebook = ctk.CTkTabview(self.root)
        self.notebook.pack(padx=20, pady=20, fill="both", expand=True)

        self.notebook.add("View Cards")
        self.notebook.add("Add Card")

        view_frame = self.notebook.tab("View Cards")
        view_button = ctk.CTkButton(view_frame, text="Refresh Bank Cards", command=self.view_all_bank_accounts)
        view_button.pack(pady=10)

        self.cards_frame = ctk.CTkScrollableFrame(view_frame, height=400)
        self.cards_frame.pack(pady=10, fill="both", expand=True)

        add_frame = self.notebook.tab("Add Card")
        self.setup_add_card_form(add_frame)

        home_button = ctk.CTkButton(view_frame, command=self.return_home, text="Return Home")
        home_button.pack()

    def delete_card(self, card_id: int):
        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this card?"):
            if Card.delete_card(card_id):
                messagebox.showinfo("Success", "Card deleted successfully!")
                self.view_all_bank_accounts()
            else:
                messagebox.showerror("Error", "Failed to delete card")

    def create_card_frame(self, parent, account_data):
        card_frame = ctk.CTkFrame(parent)
        card_frame.pack(pady=5, fill="x", expand=True)

        card_id, username, card_holder, card_number, expiry, card_type = account_data
        masked_card = f"****-****-****-{card_number[-4:]}"
        
        info_frame = ctk.CTkFrame(card_frame)
        info_frame.pack(side="left", padx=10, pady=5, fill="x", expand=True)
        
        labels = [
            f"Card ID: {card_id}",
            # f"Username: {username}",
            f"Cardholder: {card_holder}",
            f"Card Number: {masked_card}",
            f"Expiry: {expiry}",
            f"Type: {card_type}"
        ]
        
        for label_text in labels:
            ctk.CTkLabel(info_frame, text=label_text).pack(anchor="w")

        delete_btn = ctk.CTkButton(
            card_frame, text="Delete Card", command=lambda cid=card_id: self.delete_card(cid),
            fg_color="red", hover_color="darkred", width=100
        )
        delete_btn.pack(side="right", padx=10)

    def view_all_bank_accounts(self):
        try:
            for widget in self.cards_frame.winfo_children():
                widget.destroy()

            accounts = session.query(Card).join(User).all()
            
            if not accounts:
                no_cards_label = ctk.CTkLabel(self.cards_frame, text="No cards found", font=("Arial", 14))
                no_cards_label.pack(pady=20)
                return

            with open("user_id.txt", "r") as f:
                current_user_id = f.readline().strip()

            for account in accounts:
                account_data = (
                    account.cardid,
                    account.userid,
                    account.card_holder_name,
                    account.card_number,
                    account.expiration_date,
                    account.card_type
                )
                
                # if account.userid == User.get_user_id(self.current_username):
                if str(account.userid) == current_user_id:
                    self.create_card_frame(self.cards_frame, account_data)



        except SQLAlchemyError as e:
            messagebox.showerror("Database Error", f"Failed to retrieve accounts: {str(e)}")

    def setup_add_card_form(self, parent):
        form_frame = ctk.CTkFrame(parent)
        form_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        ctk.CTkLabel(form_frame, text="Username:").pack(pady=(10, 0))
        self.username_entry = ctk.CTkEntry(form_frame)
        self.username_entry.pack(pady=(0, 10))
        self.username_entry.insert(0, "james_wilson")

        ctk.CTkLabel(form_frame, text="Cardholder Name:").pack(pady=(10, 0))
        self.card_holder_entry = ctk.CTkEntry(form_frame)
        self.card_holder_entry.pack(pady=(0, 10))
        self.card_holder_entry.insert(0, "James Wilson")

        ctk.CTkLabel(form_frame, text="Card Number:").pack(pady=(10, 0))
        self.card_number_entry = ctk.CTkEntry(form_frame)
        self.card_number_entry.pack(pady=(0, 10))
        self.card_number_entry.insert(0, "4532015112830366")

        ctk.CTkLabel(form_frame, text="Expiry Date (MM/YY):").pack(pady=(10, 0))
        self.expiration_entry = ctk.CTkEntry(form_frame)
        self.expiration_entry.pack(pady=(0, 10))
        self.expiration_entry.insert(0, "12/25")

        ctk.CTkLabel(form_frame, text="Card Type:").pack(pady=(10, 0))
        self.card_type_var = ctk.StringVar(value="Visa Debit")
        card_types = ["Visa Debit", "Mastercard Debit", "American Express", "Visa Credit", "Mastercard Credit"]
        self.card_type_dropdown = ctk.CTkOptionMenu(form_frame, values=card_types, variable=self.card_type_var)
        self.card_type_dropdown.pack(pady=(0, 10))

        ctk.CTkLabel(form_frame, text="CVV:").pack(pady=(10, 0))
        self.cvv_entry = ctk.CTkEntry(form_frame, show="*")
        self.cvv_entry.pack(pady=(0, 10))
        self.cvv_entry.insert(0, "123")

        submit_btn = ctk.CTkButton(form_frame, text="Add Card", command=self.add_card)
        submit_btn.pack(pady=20)

        clear_btn = ctk.CTkButton(form_frame, text="Clear Form", command=self.clear_form)
        clear_btn.pack(pady=(0, 20))

    def clear_form(self):
        self.username_entry.delete(0, tk.END)
        self.card_holder_entry.delete(0, tk.END)
        self.card_number_entry.delete(0, tk.END)
        self.expiration_entry.delete(0, tk.END)
        self.cvv_entry.delete(0, tk.END)
        self.card_type_var.set("Visa Debit")

    def validate_card_number(self, card_number: str) -> bool:
        card_number = card_number.replace(" ", "").replace("-", "")
        if not (13 <= len(card_number) <= 19) or not card_number.isdigit():
            return False
            
        digits = [int(d) for d in card_number]
        checksum = sum(d if i % 2 != len(digits) % 2 else d * 2 - 9 * (d * 2 > 9) for i, d in enumerate(digits))
        return checksum % 10 == 0

    def validate_expiration_date(self, exp_date: str) -> bool:
        if not re.match(r"^(0[1-9]|1[0-2])/([0-9]{2})$", exp_date):
            return False
        month, year = map(int, exp_date.split("/"))
        return datetime(2000 + year, month, 1) > datetime.now()

    def validate_cvv(self, cvv: str) -> bool:
        return cvv.isdigit() and len(cvv) in (3, 4) and (self.card_type_var.get() == "American Express" and len(cvv) == 4 or len(cvv) == 3)

    def add_card(self):
        username = self.username_entry.get().strip()
        card_holder = self.card_holder_entry.get().strip()
        card_number = self.card_number_entry.get().strip().replace(" ", "").replace("-", "")
        expiration = self.expiration_entry.get().strip()
        card_type = self.card_type_var.get()
        cvv = self.cvv_entry.get().strip()

        if not username:
            messagebox.showerror("Error", "Username is required")
            return

        if not card_holder:
            messagebox.showerror("Error", "Cardholder name is required")
            return

        if not self.validate_card_number(card_number):
            messagebox.showerror("Error", "Invalid card number")
            return

        if not self.validate_expiration_date(expiration):
            messagebox.showerror("Error", "Invalid expiry date")
            return

        if not self.validate_cvv(cvv):
            messagebox.showerror("Error", "Invalid CVV")
            return

        user = User.get_user_by_username(username)
        if user is None:
            messagebox.showerror("Error", "User not found")
            return

        card = Card(
            userid=user.userid,
            card_holder_name=card_holder,
            card_number=card_number,
            expiration_date=expiration,
            card_type=card_type,
            cvv_code=cvv
        )

        if card.save_card():
            messagebox.showinfo("Success", "Card added successfully!")
            self.view_all_bank_accounts()
            self.clear_form()
        else:
            messagebox.showerror("Error", "Failed to add card")


