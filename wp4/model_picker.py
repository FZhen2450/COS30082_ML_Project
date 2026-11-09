"""Shared local-file model picker; model loading stays in PlantApp's worker."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .model_loader import MODEL_SPECS, validate_model_inputs
from .widgets import BG


class ModelPicker(tk.Toplevel):
    def __init__(self, parent, on_load, initial="wp1"):
        super().__init__(parent)
        self.on_load = on_load
        self._closed = False
        self.title("LeafLens | Connect a recognition model")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        frame = ttk.Frame(self, padding=22)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Connect a recognition model", font=("Segoe UI", -20, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(frame, text="Choose a model and its supplied local weight files.").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(8, 16))
        ttk.Label(frame, text="Model").grid(row=2, column=0, sticky="w", padx=(0, 10))
        self.kinds = list(MODEL_SPECS)
        self.choice = ttk.Combobox(frame, state="readonly", width=44,
                                  values=[spec["label"] for spec in MODEL_SPECS.values()])
        self.choice.grid(row=2, column=1, columnspan=2, sticky="ew")
        self.choice.current(self.kinds.index(initial))
        self.choice.bind("<<ComboboxSelected>>", self.render_fields)
        self.file_frame = ttk.Frame(frame)
        self.file_frame.grid(row=3, column=0, columnspan=3, pady=(12, 0))
        self.paths = {key: tk.StringVar(self) for key in ("crop", "disease", "clip_weights", "adapter_checkpoint")}
        self.device = tk.StringVar(self, value="cpu")
        ttk.Label(frame, text="Device").grid(row=4, column=0, sticky="w", pady=12)
        ttk.Combobox(frame, textvariable=self.device, values=("cpu", "cuda"), state="readonly", width=12).grid(
            row=4, column=1, sticky="w")
        ttk.Button(frame, text="Load model", style="Accent.TButton", command=self.load).grid(
            row=5, column=0, columnspan=3, sticky="e", pady=(10, 0))
        self.render_fields()

    @property
    def kind(self):
        return self.kinds[self.choice.current()]

    def render_fields(self, _event=None):
        for child in self.file_frame.winfo_children():
            child.destroy()
        for row, (key, label) in enumerate(MODEL_SPECS[self.kind]["fields"]):
            ttk.Label(self.file_frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=8)
            ttk.Entry(self.file_frame, textvariable=self.paths[key], width=44).grid(row=row, column=1)
            ttk.Button(self.file_frame, text="Browse...", command=lambda k=key: self.choose_file(k)).grid(
                row=row, column=2, padx=(8, 0))

    def choose_file(self, key):
        filename = filedialog.askopenfilename(parent=self, filetypes=[("PyTorch weights", "*.pt *.pth")])
        if filename:
            self.paths[key].set(filename)

    def load(self):
        kind, device = self.kind, self.device.get()
        paths = {key: self.paths[key].get() for key, _ in MODEL_SPECS[kind]["fields"]}
        try:
            validate_model_inputs(kind, paths, device)
        except ValueError as exc:
            messagebox.showerror("Check model files", str(exc), parent=self)
            return
        callback = self.on_load
        self.destroy()
        callback(kind, paths, device)

    def destroy(self):
        if self._closed:
            return
        self._closed = True
        self.on_load = None
        self.paths.clear()
        self.device = None
        super().destroy()
