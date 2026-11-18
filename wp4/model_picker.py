"""Inline model selection with cancellable background weight discovery."""
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .asset_discovery import discover_weight_files
from .model_loader import MODEL_SPECS, validate_model_inputs
from .widgets import WHITE, INK, MUTED, Button, label

SHORT_LABELS = {"wp1": "WP1 / ResNet50 pair", "frozen_clip": "Frozen CLIP / ViT-L/14", "m6": "M6 / CLIP adapter"}


class ModelPanel(tk.Frame):
    def __init__(self, parent, on_load, initial="wp1"):
        super().__init__(parent, bg=WHITE)
        self.on_load = on_load
        self._closed = False
        self._search_token = 0
        self._search_stop = threading.Event()
        self._search_events = queue.Queue()
        self._search_after = None
        self._details_open = False
        self._enabled = True
        self.final_training_sha = None
        self.kinds = list(MODEL_SPECS)
        self.choice = ttk.Combobox(self, state="readonly", width=18,
                                  style="Leaf.TCombobox", font=("Segoe UI", -16),
                                  values=[SHORT_LABELS[kind] for kind in self.kinds])
        self.choice.pack(fill="x", pady=(0, 8), ipady=3)
        self.choice.current(self.kinds.index(initial))
        self.choice.bind("<<ComboboxSelected>>", self.render_fields)
        self.search_status = label(self, "Finding local weights...", 12, MUTED,
                                   wraplength=230, justify="left", anchor="nw", height=3)
        self.search_status.pack(fill="x", pady=(0, 8))
        tools = tk.Frame(self, bg=WHITE)
        tools.pack(fill="x", pady=(0, 12))
        self.details_button = ttk.Button(tools, text="Files", style="Small.TButton", command=self.toggle_details)
        self.details_button.pack(side="left")
        self.search_button = ttk.Button(tools, text="Search folder", style="Small.TButton", command=self.search_folder)
        self.search_button.pack(side="right")
        self.file_frame = tk.Frame(self, bg=WHITE)
        self.paths = {key: tk.StringVar(self) for key in ("crop", "disease", "clip_weights", "adapter_checkpoint")}
        bottom = tk.Frame(self, bg=WHITE)
        bottom.pack(fill="x", pady=(0, 12))
        label(bottom, "Device", 13, MUTED).pack(side="left")
        self.device = tk.StringVar(self, value="cpu")
        self.device_choice = ttk.Combobox(bottom, textvariable=self.device, values=("cpu", "cuda"),
                                         style="Leaf.TCombobox", font=("Segoe UI", -16),
                                         state="readonly", width=7)
        self.device_choice.pack(side="right")
        self.load_button = Button(self, "Load model", self.load, height=46)
        self.load_button.pack(fill="x")
        self.render_fields()
        self._search_after = self.after(80, self.poll_search)

    @property
    def kind(self):
        return self.kinds[self.choice.current()]

    def set_protocol(self, protocol):
        from final_protocol import SCHEMA
        training_sha = protocol.get('training_manifest_sha256') if protocol.get('schema') == SCHEMA else None
        if training_sha != self.final_training_sha:
            self.final_training_sha = training_sha
            for key in ('crop', 'disease', 'adapter_checkpoint'):
                self.paths[key].set('')
            self.choice.configure(values=[SHORT_LABELS[kind] for kind in self.kinds])
            self.start_search()

    def render_fields(self, _event=None):
        for child in self.file_frame.winfo_children():
            child.destroy()
        for key, caption in MODEL_SPECS[self.kind]["fields"]:
            label(self.file_frame, caption, 11, MUTED).pack(anchor="w", pady=(4, 4))
            row = tk.Frame(self.file_frame, bg=WHITE)
            row.pack(fill="x", pady=(0, 8))
            ttk.Entry(row, textvariable=self.paths[key], width=12, font=("Segoe UI", -14)).pack(
                side="left", fill="x", expand=True)
            ttk.Button(row, text="...", width=3, style="Small.TButton",
                       command=lambda k=key: self.choose_file(k)).pack(side="right", padx=(5, 0))
        self.start_search()

    def toggle_details(self):
        self._details_open = not self._details_open
        if self._details_open:
            self.file_frame.pack(fill="x", before=self.load_button, pady=(0, 10))
        else:
            self.file_frame.pack_forget()
        self.details_button.configure(text="Hide files" if self._details_open else "Files")

    def set_enabled(self, enabled):
        self._enabled = enabled
        for control in (self.choice, self.device_choice):
            control.configure(state="readonly" if enabled else "disabled")
        for control in (self.details_button, self.search_button, self.load_button):
            control.configure(state="normal" if enabled else "disabled")

    def start_search(self, roots=None):
        self._search_stop.set()
        self._search_stop = threading.Event()
        self._search_token += 1
        token, kind = self._search_token, self.kind
        events, stop = self._search_events, self._search_stop
        final_training_sha = self.final_training_sha
        self.search_status.configure(text="Finding and verifying local weights...")

        def search():
            # Only plain data crosses the worker boundary.
            try:
                result = discover_weight_files(kind, roots=roots, stop=stop, final_training_sha=final_training_sha)
                events.put((token, kind, result, None))
            except Exception as exc:
                events.put((token, kind, None, str(exc)))

        threading.Thread(target=search, daemon=True).start()

    def poll_search(self):
        if self._closed:
            return
        while True:
            try:
                token, kind, result, error = self._search_events.get_nowait()
            except queue.Empty:
                break
            if token != self._search_token or kind != self.kind:
                continue
            if error:
                self.search_status.configure(text="Search unavailable. Use Files or Search folder.")
                continue
            for key, path in result["paths"].items():
                if not self.paths[key].get():
                    self.paths[key].set(path)
            missing = [key for key, _ in MODEL_SPECS[kind]["fields"] if not self.paths[key].get()]
            self.search_status.configure(text=("Weights ready. Select Load model." if not missing else
                                              "Weights missing. Use Files or Search folder."))
        self._search_after = self.after(80, self.poll_search)

    def search_folder(self):
        if not self._enabled:
            return
        folder = filedialog.askdirectory(parent=self.winfo_toplevel(), title="Search a folder for model weights")
        if folder:
            self.start_search([folder])

    def choose_file(self, key):
        if not self._enabled:
            return
        filename = filedialog.askopenfilename(parent=self.winfo_toplevel(), filetypes=[("PyTorch weights", "*.pt *.pth")])
        if filename:
            self.paths[key].set(filename)

    def load(self):
        if not self._enabled:
            return
        kind, device = self.kind, self.device.get()
        paths = {key: self.paths[key].get() for key, _ in MODEL_SPECS[kind]["fields"]}
        try:
            validate_model_inputs(kind, paths, device)
        except ValueError as exc:
            messagebox.showerror("Check model files", str(exc), parent=self.winfo_toplevel())
            return
        self.on_load(kind, paths, device)

    def destroy(self):
        if self._closed:
            return
        self._closed = True
        self._search_stop.set()
        if self._search_after is not None:
            self.after_cancel(self._search_after)
        self.on_load = None
        self.paths.clear()
        self.device = None
        super().destroy()
