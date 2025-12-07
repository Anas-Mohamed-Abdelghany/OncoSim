import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import threading
import requests
import json
import base64
import os
import io  # <-- ADD THIS IMPORT
from PIL import Image

class UltimateChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenRouter Vision Chat (Upload or URL)")
        self.root.geometry("800x650")

        self.api_key = self.load_api_key()
        if not self.api_key:
            self.setup_ui(enabled=False)
            return

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "amazon/nova-2-lite-v1:free"
        
        self.local_image_path = None
        
        self.setup_ui(enabled=True)

    def load_api_key(self):
        """Loads the OpenRouter API key directly from the code."""
        key = "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" # <--- PASTE YOUR OPENROUTER API KEY HERE
        
        if not key or "YourOpenRouterAPIKeyHere" in key:
            messagebox.showerror("API Key Not Set", "Please edit the script and set your OpenRouter key.")
            return None
        return key

    def setup_ui(self, enabled=True):
        image_frame = tk.Frame(self.root, pady=5)
        image_frame.pack(fill=tk.X, padx=10, pady=5)

        url_row = tk.Frame(image_frame)
        url_row.pack(fill=tk.X, pady=2)
        tk.Label(url_row, text="Image URL:", font=("Arial", 11), width=15, anchor='w').pack(side=tk.LEFT)
        self.image_url_entry = tk.Entry(url_row, font=("Arial", 11))
        self.image_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        file_row = tk.Frame(image_frame)
        file_row.pack(fill=tk.X, pady=2)
        self.upload_button = tk.Button(file_row, text="Upload From Device...", command=self.upload_local_file)
        self.upload_button.pack(side=tk.LEFT)
        self.status_label = tk.Label(file_row, text="No file selected.", fg="gray", anchor='w')
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.clear_button = tk.Button(file_row, text="Clear Image", command=self.clear_selection)
        self.clear_button.pack(side=tk.RIGHT)

        chat_frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        chat_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state='disabled', font=("Arial", 11))
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.add_message("System: Upload an image or paste a URL, then ask a question.", "system")
        
        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=10, fill=tk.X)
        self.entry_box = tk.Entry(input_frame, font=("Arial", 11))
        self.entry_box.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.entry_box.bind("<Return>", self.send_message)
        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))

        if not enabled:
            # Disable controls if setup fails
            for widget in [self.image_url_entry, self.upload_button, self.clear_button, self.entry_box, self.send_button]:
                widget.config(state='disabled')
            self.add_message("System: Application disabled. Please set API key.", "error")

    def upload_local_file(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")])
        if path:
            self.local_image_path = path
            self.status_label.config(text=f"Selected: {os.path.basename(path)}", fg="blue")
            self.image_url_entry.delete(0, tk.END)

    def clear_selection(self):
        self.local_image_path = None
        self.image_url_entry.delete(0, tk.END)
        self.status_label.config(text="No file selected.", fg="gray")

    # --- THIS IS THE FIXED FUNCTION ---
    def encode_image_to_data_url(self, path):
        """
        Opens any image, converts it to PNG format in memory,
        and returns a Base64 data URL. This ensures compatibility.
        """
        try:
            with Image.open(path) as img:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
            
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/png;base64,{base64_string}"
        except Exception as e:
            print(f"Error converting image to PNG: {e}")
            raise e

    def send_message(self, event=None):
        user_prompt = self.entry_box.get().strip()
        image_url = self.image_url_entry.get().strip()
        
        if not user_prompt:
            messagebox.showwarning("Input Required", "Please type a question.")
            return

        self.add_message(f"You: {user_prompt}", "user")
        if self.local_image_path:
            self.add_message(f"(With uploaded image: {os.path.basename(self.local_image_path)})", "system")
        elif image_url:
            self.add_message(f"(With image from URL)", "system")
        
        self.entry_box.delete(0, tk.END)
        self.enable_controls(False)
        
        threading.Thread(target=self.get_bot_response, args=(user_prompt, image_url, self.local_image_path)).start()

    def get_bot_response(self, user_prompt, image_url, local_path):
        headers = { "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json" }
        user_content = [{"type": "text", "text": user_prompt}]
        final_image_url = None

        if local_path:
            try:
                final_image_url = self.encode_image_to_data_url(local_path)
            except Exception as e:
                self.root.after(0, self.add_message, f"Error encoding image: {e}", "error")
                self.root.after(0, self.enable_controls, True)
                return
        elif image_url:
            final_image_url = image_url

        if final_image_url:
            user_content.append({"type": "image_url", "image_url": {"url": final_image_url}})

        api_messages = [{"role": "user", "content": user_content}]
        payload = { "model": self.model, "messages": api_messages, "max_tokens": 2048 }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            bot_reply = response_data['choices'][0]['message']['content']
            self.root.after(0, self.add_message, f"Bot: {bot_reply}", "bot")
        except requests.exceptions.RequestException as e:
            self.root.after(0, self.add_message, f"Error: {e}", "error")
        finally:
            self.root.after(0, self.enable_controls, True)
            self.root.after(0, self.clear_selection)

    def enable_controls(self, is_enabled):
        state = 'normal' if is_enabled else 'disabled'
        for widget in [self.entry_box, self.send_button, self.image_url_entry, self.upload_button, self.clear_button]:
            widget.config(state=state)
        if is_enabled:
            self.entry_box.focus_set()

    def add_message(self, message, tag=None):
        self.chat_display.config(state='normal')
        self.chat_display.tag_config("user", foreground="black", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("bot", foreground="green")
        self.chat_display.tag_config("error", foreground="red")
        self.chat_display.tag_config("system", foreground="gray", font=("Arial", 10, "italic"))
        self.chat_display.insert(tk.END, message + "\n\n", tag)
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = UltimateChatApp(root)
    root.mainloop()