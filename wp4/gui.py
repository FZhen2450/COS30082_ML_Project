"""English Tkinter desktop GUI; all image/model work runs off the UI thread."""
import argparse
import json
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from .service import Catalog, load_independent, open_rgb, present_prediction
from .widgets import BG, WHITE, INK, MUTED, GREEN, Card, label, rounded

ROOT = Path(__file__).resolve().parents[1]


class PlantApp:
    def __init__(self, root, protocol, images, source):
        self.root = root
        self.protocol_path, self.images_path, self.source_path = protocol, images, source
        self.catalog = self.predictor = self.query = self.query_path = None
        self.predictions = []
        self.export_data = None
        self.photos = []
        self.events = queue.Queue()
        self.busy = False
        self.closed = False
        self.model_label = "No model loaded"
        self.root.title("LeafLens | Plant disease identification")
        width = min(1240, self.root.winfo_screenwidth() - 80)
        height = min(940, self.root.winfo_screenheight() - 100)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1100, 800)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._style()
        self._build()
        self.root.after(70, self._poll)
        self.root.after(100, self.load_catalog)

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", -13))
        style.configure("TButton", background=WHITE, foreground=INK, borderwidth=0,
                        font=("Segoe UI", -13), padding=(14, 9))
        style.map("TButton", background=[("active", "#eaf1e7")])
        style.configure("Accent.TButton", background=GREEN, foreground=WHITE)
        style.map("Accent.TButton", background=[("active", "#194f38")])
        style.configure("Leaf.TCombobox", fieldbackground="#f7f9f5", background="#f7f9f5",
                        foreground=INK, arrowcolor=MUTED, bordercolor="#e5eae3",
                        lightcolor="#e5eae3", darkcolor="#e5eae3", borderwidth=1,
                        font=("Segoe UI", -12), padding=(9, 3))
        style.map("Leaf.TCombobox", fieldbackground=[("readonly", "#f7f9f5")],
                  selectbackground=[("readonly", "#f7f9f5")], selectforeground=[("readonly", INK)])
        style.configure("Leaf.Vertical.TScrollbar", troughcolor=WHITE, background="#dce4d8",
                        borderwidth=0, arrowsize=10, arrowcolor=MUTED)
        style.configure("Leaf.Horizontal.TProgressbar", troughcolor="#e8ede4",
                        background="#8ca486", borderwidth=0, thickness=3)
        self.root.option_add("*TCombobox*Listbox.font", ("Segoe UI", -12))
        self.root.option_add("*TCombobox*Listbox.background", WHITE)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#eaf3e8")
        self.root.option_add("*TCombobox*Listbox.selectForeground", INK)

    def _build(self):
        from .layout import build
        build(self)

    def _draw_query(self):
        canvas = self.image_panel
        w, h = canvas.winfo_width(), canvas.winfo_height()
        canvas.delete("all")
        rounded(canvas, (0, 0, w, h), 16, fill="#f1f4ec", outline="")
        if self.query is not None:
            preview = self.query.copy()
            preview.thumbnail((max(1, w-20), max(1, h-20)))
            self.query_photo = ImageTk.PhotoImage(preview)
            canvas.create_image(w/2, h/2, image=self.query_photo)
        else:
            cx, cy = w/2, h/2-17
            canvas.create_oval(cx-26, cy-26, cx+26, cy+26, fill="#e5eddd", outline="")
            canvas.create_polygon(cx-11, cy+14, cx-17, cy-10, cx+15, cy-16,
                                  cx+19, cy+3, cx-11, cy+14, smooth=True, fill="#a0bb94", outline="")
            canvas.create_line(cx-12, cy+14, cx+11, cy-10, fill=GREEN, width=1.5)
            canvas.create_text(cx, cy+48, text="A leaf. A new perspective.", fill=INK, font=("Segoe UI", -12))
            canvas.create_text(cx, cy+68, text="JPG, PNG, BMP or WEBP", fill=MUTED, font=("Segoe UI", -10))

    def _empty_gallery(self):
        box = tk.Frame(self.gallery, bg=WHITE)
        box.grid(row=0, column=0, columnspan=2, sticky="ew", pady=24)
        label(box, "Make a connection", 15, bold=True).pack()
        label(box, "Related training images will appear here.", 12, MUTED).pack(pady=6)

    def _controls(self):
        available = not self.busy
        self.table.enabled = available
        for widget in (self.open_button, self.configure_button):
            widget.configure(state="normal" if available else "disabled")
        self.identify_button.configure(state="normal" if available and self.predictor and self.query else "disabled")
        self.browse_button.configure(state="normal" if available and self.catalog else "disabled")
        self.class_choice.configure(state="readonly" if available and self.catalog else "disabled")
        self.export_button.configure(state="normal" if available and self.export_data else "disabled")

    def _work(self, action, done, description):
        if self.busy:
            return
        self.busy = True
        self.status.configure(text=description)
        self.progress.start(12)
        self._controls()
        def run():
            try:
                self.events.put((done, action(), None))
            except Exception as exc:
                self.events.put((done, None, str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        if self.closed:
            return
        try:
            done, value, error = self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.progress.stop()
            self.status.configure(text="Ready" if not error else "Action could not be completed")
            self._controls()
            if error:
                messagebox.showerror("Unable to complete action", error, parent=self.root)
            else:
                done(value)
        self.root.after(70, self._poll)

    def load_catalog(self):
        def done(catalog):
            self.catalog = catalog
            self.class_ids = list(catalog.candidates)
            self.class_choice.configure(values=[f"{c['crop_name']} / {c['disease_name']}"
                                                for cid, c in catalog.candidates.items()])
            self.class_choice.current(0)
            self.status.configure(text=f"{len(catalog.fitting):,} fitting references | {len(catalog.candidates)} candidates | Model not loaded")
            self._controls()
        self._work(lambda: Catalog(self.protocol_path, self.source_path, self.images_path), done,
                   "Checking the training reference catalog...")

    def open_image(self):
        filename = filedialog.askopenfilename(parent=self.root, title="Choose a plant image",
                    filetypes=[("Plant images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if filename:
            def done(image):
                self.query, self.query_path = image, Path(filename)
                self._draw_query()
                self.file_label.configure(text=f"{Path(filename).name}\n{image.width} x {image.height} pixels")
                self.clear_predictions()
                self._controls()
            self._work(lambda: open_rgb(filename), done, "Opening image...")

    def clear_predictions(self):
        self.table.delete(*self.table.get_children())
        self.predictions, self.export_data = [], None
        self.result_note.configure(text="Image ready. Connect a model, then select Analyze leaf.")
        for widget in self.gallery.winfo_children():
            widget.destroy()
        self.photos = []
        self.reference_note.configure(text="Choose a match above or explore a labelled class.")
        self._empty_gallery()

    def configure_model(self):
        if not self.catalog:
            messagebox.showinfo("Catalog not ready", "Start with a valid protocol and training CSV. See docs/WP4_GUI.md.")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("LeafLens | Connect a recognition model")
        dialog.configure(bg=BG)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        frame = ttk.Frame(dialog, padding=22)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Connect a recognition model", font=("Segoe UI", -20, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(frame, text="Choose two trained .pt checkpoints. Model loading runs locally.").grid(row=1, column=0, columnspan=3, pady=(8, 18), sticky="w")
        crop, disease, device = tk.StringVar(), tk.StringVar(), tk.StringVar(value="cpu")
        for index, (label, variable) in enumerate((("Crop checkpoint", crop), ("Disease checkpoint", disease)), 2):
            ttk.Label(frame, text=label).grid(row=index, column=0, sticky="w", padx=(0, 10), pady=8)
            ttk.Entry(frame, textvariable=variable, width=44, font=("Segoe UI", -12)).grid(row=index, column=1)
            def choose(var=variable):
                name = filedialog.askopenfilename(parent=dialog, filetypes=[("PyTorch checkpoints", "*.pt *.pth")])
                if name:
                    var.set(name)
            ttk.Button(frame, text="Browse...", command=choose).grid(row=index, column=2, padx=(8, 0))
        ttk.Label(frame, text="Device").grid(row=4, column=0, sticky="w", pady=10)
        ttk.Combobox(frame, textvariable=device, values=("cpu", "cuda"), state="readonly", width=12).grid(row=4, column=1, sticky="w")
        def load():
            cp, dp, dev = crop.get(), disease.get(), device.get()
            if not Path(cp).is_file() or not Path(dp).is_file():
                messagebox.showerror("Check checkpoint paths", "Select both existing checkpoint files.", parent=dialog)
                return
            dialog.destroy()
            def done(predictor):
                self.predictor = predictor
                self.model_label = f"WP1 independent classifiers ({dev.upper()})"
                self.model_status.configure(text=self.model_label)
                self.clear_predictions()
                self._controls()
            self._work(lambda: load_independent(self.catalog, cp, dp, dev), done, "Loading trained checkpoints...")
        ttk.Button(frame, text="Load model", style="Accent.TButton", command=load).grid(row=5, column=0, columnspan=3, sticky="e", pady=(16, 0))

    def identify(self):
        if not self.predictor or self.query is None:
            return
        def action():
            started = time.perf_counter()
            raw = self.predictor.predict([self.query], topk=5)
            return present_prediction(raw, self.catalog), raw["score_semantics"], time.perf_counter() - started
        def done(result):
            predictions, semantics, elapsed = result
            self.table.delete(*self.table.get_children())
            self.predictions = predictions
            for index, row in enumerate(predictions):
                self.table.insert("", "end", iid=str(row["composition_id"]),
                                  values=(index + 1, row["crop_name"], row["disease_name"], f"{row['score']:.4f}"))
            self.result_note.configure(text=f"Top {len(predictions)} crop-disease matches | {elapsed:.2f} s | {self.model_label}")
            self.export_data = {"image": self.query_path.name, "model": self.model_label,
                                "scope": "development candidates; not official-test evaluation",
                                "candidate_sha256": self.catalog.protocol["candidate_sha256"],
                                "split_sha256": self.catalog.protocol["split_sha256"],
                                "historical_reconciliation": self.catalog.protocol["historical_reconciliation"],
                                "score_semantics": semantics, "elapsed_seconds": elapsed,
                                "predictions": predictions}
            self._controls()
            self.table.selection_set(str(predictions[0]["composition_id"]))
        self._work(action, done, "Identifying crop and disease...")

    def select_prediction(self, _event=None):
        selection = self.table.selection()
        if selection and not self.busy:
            self.show_references(int(selection[0]), True)

    def browse_class(self):
        index = self.class_choice.current()
        if index >= 0 and self.catalog:
            self.show_references(self.class_ids[index], False)

    def show_references(self, composition_id, from_prediction):
        if composition_id in self.class_ids:
            self.class_choice.current(self.class_ids.index(composition_id))
        query = self.query if from_prediction else None
        def done(result):
            for widget in self.gallery.winfo_children():
                widget.destroy()
            self.photos = []
            self.gallery_canvas.yview_moveto(0)
            origin = "Selected match" if from_prediction else "Class browser (not a prediction)"
            method = "Closest colours first." if result["query_ordered"] else "Filename order."
            note = f"{origin}. {result['relation']}. Fitting images only. {method}"
            if result["unavailable"]:
                note += f" {result['unavailable']} missing or unreadable; check your image folder."
            self.reference_note.configure(text=note)
            for index, item in enumerate(result["items"]):
                card = Card(self.gallery, height=206, padding=10, fill="#f7f9f5")
                card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=4, pady=(0, 8))
                preview = item["image"].copy()
                preview.thumbnail((220, 122))
                photo = ImageTk.PhotoImage(preview)
                self.photos.append(photo)
                area = tk.Canvas(card.body, height=122, bg="#edf1e8", highlightthickness=0)
                area.pack(fill="x")
                photo_id = area.create_image(110, 61, image=photo)
                area.bind("<Configure>", lambda e, a=area, i=photo_id: a.coords(i, e.width/2, 61))
                label(card.body, item["disease"].replace("_", " ").capitalize(), 12, bold=True,
                      anchor="w", wraplength=220).pack(fill="x", pady=(8, 2))
                label(card.body, item["crop"].replace("_", " ").capitalize(), 11, MUTED,
                      anchor="w").pack(fill="x")
                name = item["filename"]
                label(card.body, name if len(name) <= 35 else name[:32]+"...", 9, MUTED,
                      anchor="w").pack(fill="x", pady=(3, 0))
            if not result["items"]:
                label(self.gallery, "No fitting references available.\nCheck the image folder or try another class.",
                      12, MUTED, justify="center").grid(row=0, column=0, columnspan=2, pady=35)
        self._work(lambda: self.catalog.references(composition_id, query, self.query_path if query is not None else None),
                   done, "Finding related training images...")

    def export(self):
        if self.export_data:
            filename = filedialog.asksaveasfilename(parent=self.root, title="Export prediction", defaultextension=".json",
                                                    initialfile="plant_prediction.json", filetypes=[("JSON", "*.json")])
            if filename:
                try:
                    Path(filename).write_text(json.dumps(self.export_data, indent=2) + "\n", encoding="utf-8")
                    self.status.configure(text="Prediction exported")
                except OSError as exc:
                    messagebox.showerror("Export failed", str(exc), parent=self.root)

    def close(self):
        self.closed = True
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    reconciled = ROOT / "artifacts/data/development_seed42_reconciled.json"
    parser.add_argument("--protocol", type=Path, default=reconciled if reconciled.exists() else ROOT / "artifacts/data/development_seed42_wp1_v1.json")
    parser.add_argument("--images", type=Path, default=ROOT / "DPD_Small_dataset")
    parser.add_argument("--source", type=Path, default=ROOT / "DPD_Small_train.csv")
    args = parser.parse_args()
    # Keep Tk window coordinates consistent with Windows display scaling.
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    PlantApp(root, args.protocol, args.images, args.source)
    root.mainloop()
