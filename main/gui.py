import tkinter as tk
import os
from tkinter import filedialog, messagebox, ttk
from blur_faces import main as blur_faces
from types import SimpleNamespace
from tkinter.scrolledtext import ScrolledText
import sys
import threading
import queue
import time

def worker_thread(args, result_queue):
    try:
        blur_faces(args)
        result_queue.put(True)
    except Exception as e:
        print("Error blurring file ", input, e)
        result_queue.put(False)

def run_blur_faces(args): 
    # Create a queue to receive the result
    result_queue = queue.Queue()
    # Need to run the blur_faces in a separate thread to prevent freezing of the GUI
    thread = threading.Thread(target=worker_thread, args=(args, result_queue))
    thread.start()
    
    # Wait loop that keeps GUI responsive
    # This is a bit of a hack, ideally, we'd use callback based notification of thread completion but this makes it easier
    while thread.is_alive():
        root.update()  # Process events, keeps GUI responsive
        time.sleep(0.01)  # Sleep a little to avoid hogging CPU

    # Retrieve result
    status = result_queue.get()

    return status

# Dummy processing function (you should replace this)
def process(input_files, output_folder, model, method, status_callback):
    print("Processing files:")
    for input in input_files:
        basename, ext = os.path.splitext(os.path.basename(input))
        output = os.path.join(output_folder, basename + "-blurred" + ext)
        args = SimpleNamespace(input = input, output = output, method = method, model = model)
        print("\n====== Blurring '" + input + "' to '" + output + "' ========")
        success = run_blur_faces(args)
        status_callback(input, "✓" if success else "✗")
    messagebox.showinfo("Done", f"Processed {len(input_files)} file(s) into:\n{output_folder}")

# Redirect class
class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ''
        self.cr = False

    def write(self, s):
        self.text_widget.configure(state='normal')
        for char in s:
            if char == '\r':
                self.cr = True
            else:
                if self.cr:
                    # Delete the last line
                    self.text_widget.delete("end-1c linestart", "end")
                    self.text_widget.insert("end-1l", "\n")  # Ensure new line context
                    self.text_widget.mark_set("insert", "end-1l")
                self.cr = False
                self.text_widget.insert("end", char)
        self.text_widget.see(tk.END)  # Auto-scroll
        self.text_widget.configure(state='disabled')

    def flush(self):  # Needed for compatibility with sys.stdout
        pass

# GUI setup
class BlurApp:
    def __init__(self, root):
        self.root = root
        root.title("Blur Daddy")
        root.geometry("700x600")
        self.input_files = []
        self.output_folder = ""
        self.buttons = []  # To track all buttons

        # Input selection
        btn_input = tk.Button(root, text="Select Input File(s)", command=self.select_input_files)
        btn_input.pack(pady=5)
        self.input_listbox = tk.Listbox(root, width=80)
        self.input_listbox.pack(pady=5)
        self.buttons.append(btn_input)

        # Output folder
        btn_output = tk.Button(root, text="Select Output Folder", command=self.select_output_folder)
        btn_output.pack(pady=5)
        self.output_label = tk.Label(root, text="No output folder selected")
        self.output_label.pack()
        self.buttons.append(btn_output)

        # Model dropdown
        tk.Label(root, text="Face Detection Model:").pack(pady=2)
        self.model_var = tk.StringVar(value="mtcnn")
        self.model_combo = ttk.Combobox(root, textvariable=self.model_var, values=["yolov8n-face", "mtcnn"], state="readonly")
        self.model_combo.pack()

        # Method dropdown
        tk.Label(root, text="Blur Method:").pack(pady=2)
        self.method_var = tk.StringVar(value="elliptical")
        self.method_combo = ttk.Combobox(root, textvariable=self.method_var, values=["gaussian", "elliptical", "pixelation"], state="readonly")
        self.method_combo.pack()

        # Blur button
        self.blur_button = tk.Button(root, text="Blur", command=self.run_process, bg="lightblue")
        self.blur_button.pack(pady=20)
        self.buttons.append(self.blur_button)

        # Output log area
        self.log_output = ScrolledText(root, wrap='word', width=80, height=20, state='disabled', font=("Consolas", 10))
        self.log_output.pack(fill="both", expand=True, padx=10, pady=10)

        # Redirect stdout
        sys.stdout = TextRedirector(self.log_output)
        sys.stderr = TextRedirector(self.log_output)

    def select_input_files(self):
        files = filedialog.askopenfilenames(title="Select input files")
        if files:
            self.input_files = list(files)
            self.refresh_file_list()

    def refresh_file_list(self):
        self.input_listbox.delete(0, tk.END)
        for file in self.input_files:
            filename = os.path.basename(file)
            self.input_listbox.insert(tk.END, f"[ ] {filename}")

    def mark_file_status(self, file, status_symbol):
        filename = os.path.basename(file)
        for i in range(self.input_listbox.size()):
            item_text = self.input_listbox.get(i)
            if filename in item_text:
                self.input_listbox.delete(i)
                self.input_listbox.insert(i, f"[{status_symbol}] {filename}")
                break

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_folder = folder
            self.output_label.config(text=self.output_folder)

    def set_buttons_enabled(self, enabled: bool):
        state = 'normal' if enabled else 'disabled'
        for btn in self.buttons:
            btn.config(state=state)

    def run_process(self):
        if not self.input_files:
            messagebox.showerror("Error", "Please select input files.")
            return
        if not self.output_folder:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        model = self.model_var.get()
        method = self.method_var.get()

        # Clear previous status indicators
        self.refresh_file_list()

        self.set_buttons_enabled(False)
        process(self.input_files, self.output_folder, model, method, self.mark_file_status)
        self.set_buttons_enabled(True)

if __name__ == "__main__":
    root = tk.Tk()
    app = BlurApp(root)
    root.mainloop()