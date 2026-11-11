"""English Tkinter desktop GUI; all image/model work runs off the UI thread."""
import argparse
import gc
import hashlib
import json
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from .service import Catalog, open_rgb, present_prediction
from .model_loader import MODEL_SPECS, load_model
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
        self._on_done = None
        self.busy = False
        self.closed = False
        self.model_label = "No model loaded"
        self.model_kind = "wp1"
        self.root.title("LeafLens | Plant disease identification")
        width = min(1680, self.root.winfo_screenwidth() - 48)
        height = min(1080, self.root.winfo_screenheight() - 80)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(1280, self.root.winfo_screenwidth()-48),
                          min(840, self.root.winfo_screenheight()-80))
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._style()
        self._build()
        if sys.platform == "win32" and self.root.state() != "withdrawn":
            self.root.state("zoomed")
        self._poll_after = self.root.after(70, self._poll)
        self._catalog_after = self.root.after(100, self.load_catalog)

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", -18))
        style.configure("TEntry", font=("Segoe UI", -18), padding=8)
        style.configure("TCombobox", font=("Segoe UI", -18), padding=8)
        style.configure("TButton", background=WHITE, foreground=INK, borderwidth=0,
                        font=("Segoe UI", -18), padding=(18, 12))
        style.map("TButton", background=[("active", "#eaf1e7")])
        style.configure("Small.TButton", font=("Segoe UI", -14), padding=(9, 7))
        style.configure("Accent.TButton", background=GREEN, foreground=WHITE)
        style.map("Accent.TButton", background=[("active", "#194f38")])
        style.configure("Leaf.TCombobox", fieldbackground="#f7f9f5", background="#f7f9f5",
                        foreground=INK, arrowcolor=MUTED, bordercolor="#e5eae3",
                        lightcolor="#e5eae3", darkcolor="#e5eae3", borderwidth=1,
                        font=("Segoe UI", -18), padding=(12, 6))
        style.map("Leaf.TCombobox", fieldbackground=[("readonly", "#f7f9f5")],
                  selectbackground=[("readonly", "#f7f9f5")], selectforeground=[("readonly", INK)])
        style.configure("Leaf.Vertical.TScrollbar", troughcolor=WHITE, background="#dce4d8",
                        borderwidth=0, arrowsize=18, arrowcolor=MUTED)
        style.configure("Leaf.Horizontal.TProgressbar", troughcolor="#e8ede4",
                        background="#8ca486", borderwidth=0, thickness=3)
        self.root.option_add("*TCombobox*Listbox.font", ("Segoe UI", -18))
        self.root.option_add("*TCombobox.font", ("Segoe UI", -18))
        self.root.option_add("*TEntry.font", ("Segoe UI", -18))
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
            size = (max(1, w), max(1, h))
            preview = ImageOps.contain(self.query, size, method=Image.Resampling.LANCZOS)
            self.query_photo = ImageTk.PhotoImage(preview)
            canvas.create_image(w/2, h/2, image=self.query_photo)
        else:
            cx, cy = w/2, h/2-17
            canvas.create_oval(cx-26, cy-26, cx+26, cy+26, fill="#e5eddd", outline="")
            canvas.create_polygon(cx-11, cy+14, cx-17, cy-10, cx+15, cy-16,
                                  cx+19, cy+3, cx-11, cy+14, smooth=True, fill="#a0bb94", outline="")
            canvas.create_line(cx-12, cy+14, cx+11, cy-10, fill=GREEN, width=1.5)
            canvas.create_text(cx, cy+48, text="A leaf. A new perspective.", fill=INK, font=("Segoe UI", -18))
            canvas.create_text(cx, cy+76, text="JPG, PNG, BMP or WEBP", fill=MUTED, font=("Segoe UI", -15))

    def _empty_gallery(self):
        box = tk.Frame(self.gallery, bg=WHITE)
        box.grid(row=0, column=0, columnspan=self.gallery_columns, sticky="ew", pady=24)
        label(box, "Make a connection", 15, bold=True).pack()
        label(box, "Related training images will appear here.", 12, MUTED,
              wraplength=250, justify="center").pack(pady=6)

    def _refresh_workspace(self):
        previous_top = self.page_canvas.canvasy(0)
        # A canvas window scrolled fully out of view can defer Configure events.
        # Refresh its requested bounds explicitly before restoring the viewport.
        self.page_canvas.update_idletasks()
        bounds = self.page_canvas.bbox("all")
        if bounds:
            self.page_canvas.configure(scrollregion=bounds)
            height = max(1, bounds[3] - bounds[1])
            top = min(previous_top, max(0, height - self.page_canvas.winfo_height()))
            self.page_canvas.yview_moveto(max(0, top) / height)

    def _controls(self):
        available = not self.busy
        self.table.enabled = available
        for widget in (self.open_button, self.configure_button):
            widget.configure(state="normal" if available else "disabled")
        self.model_panel.set_enabled(available and self.catalog is not None)
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
        self._on_done = done
        events = self.events
        def run():
            try:
                events.put((action(), None))
            except Exception as exc:
                events.put((None, str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        if self.closed:
            return
        try:
            value, error = self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            done, self._on_done = self._on_done, None
            self.busy = False
            self.progress.stop()
            self.status.configure(text="Ready" if not error else "Action could not be completed")
            self._controls()
            if error:
                messagebox.showerror("Unable to complete action", error, parent=self.root)
            else:
                done(value)
        self._poll_after = self.root.after(70, self._poll)

    def load_catalog(self):
        protocol, source, images = self.protocol_path, self.source_path, self.images_path
        def done(catalog):
            self.catalog = catalog
            self.class_ids = list(catalog.candidates)
            self.class_choice.configure(values=[f"{c['crop_name']} / {c['disease_name']}"
                                                for cid, c in catalog.candidates.items()])
            self.class_choice.current(0)
            self.status.configure(text=f"{len(catalog.fitting):,} fitting references | {len(catalog.candidates)} candidates | Model not loaded")
            self._controls()
        self._work(lambda: Catalog(protocol, source, images), done,
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
                self.page_canvas.yview_moveto(0)
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
        self._refresh_workspace()

    def configure_model(self):
        if not self.catalog:
            messagebox.showinfo("Catalog not ready", "Start with a valid protocol and training CSV. See docs/WP4_GUI.md.")
            return
        self.model_panel.choice.focus_set()

    def connect_model(self, kind, paths, device):
        paths = dict(paths)
        catalog = self.catalog
        def done(predictor):
            self.predictor = predictor
            self.model_kind = kind
            self.model_label = f"{MODEL_SPECS[kind]['label']} ({device.upper()})"
            self.model_status.configure(text=self.model_label)
            self.clear_predictions()
            self._controls()
        self._work(lambda: load_model(catalog, kind, paths, device), done, "Loading local recognition model...")

    def identify(self):
        if not self.predictor or self.query is None:
            return
        predictor, query, catalog, model_kind = self.predictor, self.query, self.catalog, self.model_kind
        def action():
            started = time.perf_counter()
            raw = predictor.predict([query], topk=5)
            return (present_prediction(raw, catalog), raw["score_semantics"],
                    time.perf_counter() - started,
                    raw.get("model_provenance", getattr(predictor, "provenance", {"variant": model_kind})))
        def done(result):
            predictions, semantics, elapsed, provenance = result
            self.table.delete(*self.table.get_children())
            self.predictions = predictions
            for index, row in enumerate(predictions):
                self.table.insert("", "end", iid=str(row["composition_id"]),
                                  values=(index + 1, row["crop_name"], row["disease_name"], f"{row['score']:.4f}"))
            from .model_picker import SHORT_LABELS
            self.result_note.configure(text=f"Top {len(predictions)} | {SHORT_LABELS[self.model_kind]} | {elapsed:.2f} s")
            self.export_data = {"image": self.query_path.name, "model": self.model_label,
                                "scope": "development candidates; not official-test evaluation",
                                "candidate_sha256": self.catalog.protocol["candidate_sha256"],
                                "split_sha256": self.catalog.protocol["split_sha256"],
                                "historical_reconciliation": self.catalog.protocol["historical_reconciliation"],
                                "score_semantics": semantics, "elapsed_seconds": elapsed,
                                "input_policy": {"decode": "PIL EXIF transpose then RGB", "batch_size": 1,
                                                 "size": list(self.query.size),
                                                 "rgb_pixel_sha256": hashlib.sha256(self.query.tobytes()).hexdigest()},
                                "model_provenance": provenance,
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
        catalog = self.catalog
        excluded = self.query_path if query is not None else None
        def done(result):
            for widget in self.gallery.winfo_children():
                widget.destroy()
            self.photos = []
            origin = "Selected match" if from_prediction else "Class browser (not a prediction)"
            method = "Closest colours first." if result["query_ordered"] else "Filename order."
            note = f"{origin}. {result['relation']}. Fitting images only. {method}"
            if result["unavailable"]:
                note += f" {result['unavailable']} missing or unreadable; check your image folder."
            self.reference_note.configure(text=note)
            for index, item in enumerate(result["items"]):
                columns = self.gallery_columns
                card = Card(self.gallery, padding=12, fill="#f0f5ec", auto_height=True)
                card.grid(row=index // columns, column=index % columns, sticky="new", padx=2, pady=(0, 12))
                preview = item["image"].copy()
                preview.thumbnail((245, 150))
                photo = ImageTk.PhotoImage(preview)
                self.photos.append(photo)
                area = tk.Canvas(card.body, height=150, bg="#e5eee0", highlightthickness=0)
                area.pack(fill="x")
                photo_id = area.create_image(122, 75, image=photo)
                area.bind("<Configure>", lambda e, a=area, i=photo_id: a.coords(i, e.width/2, 75))
                disease_label = label(card.body, item["disease"].replace("_", " ").capitalize(), 14, bold=True,
                                      anchor="w", justify="left", wraplength=340)
                disease_label.pack(fill="x", pady=(12, 6))
                label(card.body, item["crop"].replace("_", " ").capitalize(), 14, MUTED,
                      anchor="w").pack(fill="x")
                name = item["filename"]
                filename_label = label(card.body, name, 12, MUTED, anchor="w", justify="left", wraplength=340)
                filename_label.pack(fill="x", pady=(6, 0))
                card.body.bind("<Configure>", lambda e, d=disease_label, f=filename_label: (
                    d.configure(wraplength=max(180, e.width)),
                    f.configure(wraplength=max(180, e.width))), add="+")
            if not result["items"]:
                label(self.gallery, "No fitting references available.\nCheck the image folder or try another class.",
                      12, MUTED, justify="center").grid(row=0, column=0, columnspan=2, pady=35)
            self._refresh_workspace()
        self._work(lambda: catalog.references(composition_id, query, excluded),
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
        if self.closed:
            return
        self.closed = True
        self._on_done = None
        self.progress.stop()
        for identifier in (self._poll_after, self._catalog_after):
            try:
                self.root.after_cancel(identifier)
            except tk.TclError:
                pass
        dialog = getattr(self, 'model_dialog', None)
        if dialog is not None:
            dialog.destroy()
            self.model_dialog = None
        self.photos.clear()
        self.query_photo = None
        self.root.destroy()
        # Destroy Tk cycles on their owning thread before later ML workers can
        # trigger a collection. Worker actions/queues contain no UI callbacks.
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    candidates = (ROOT / "artifacts/data/development_seed42_reconciled.json",
                  ROOT / "development_seed42_reconciled.json",
                  ROOT / "artifacts/data/development_seed42_reconciled_v1.json")
    parser.add_argument("--protocol", type=Path, default=next((p for p in candidates if p.is_file()), candidates[-1]))
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
