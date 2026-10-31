"""Responsive layout and restrained botanical styling for the desktop app."""
import tkinter as tk
from tkinter import ttk

from .widgets import BG, WHITE, INK, MUTED, GREEN, LINE, Button, Card, RankedList, label, rounded


def section(parent, eyebrow, title, trailing=None):
    line = tk.Frame(parent, bg=parent.cget("bg"))
    line.pack(fill="x")
    label(line, eyebrow, 10, MUTED, bold=True).pack(anchor="w")
    title_line = tk.Frame(line, bg=parent.cget("bg"))
    title_line.pack(fill="x", pady=(5, 0))
    label(title_line, title, 19, bold=True).pack(side="left")
    if trailing:
        label(title_line, trailing, 11, GREEN).pack(side="right")


def build(app):
    outer = tk.Frame(app.root, bg=BG, padx=28, pady=22)
    outer.pack(fill="both", expand=True)
    header = tk.Frame(outer, bg=BG)
    header.pack(fill="x", pady=(0, 20))
    logo = tk.Canvas(header, width=34, height=34, bg=BG, highlightthickness=0)
    logo.pack(side="left", padx=(0, 10))
    rounded(logo, (0, 0, 34, 34), 10, fill=GREEN, outline="")
    logo.create_polygon(10, 26, 7, 12, 25, 8, 28, 20, 10, 26, smooth=True, fill="#dceacb", outline="")
    logo.create_line(10, 25, 23, 12, fill=GREEN, width=1.5)
    label(header, "LeafLens", 23, bold=True).pack(side="left")
    label(header, "  /  Plant intelligence", 12, MUTED).pack(side="left", padx=12)
    chip = tk.Frame(header, bg="#e7eee3", padx=12, pady=7)
    chip.pack(side="right")
    label(chip, "LOCAL WORKSPACE", 10, GREEN, bold=True).pack()
    intro = tk.Frame(outer, bg=BG)
    intro.pack(fill="x", pady=(0, 20))
    label(intro, "A little insight. A healthier leaf.", 29, bold=True).pack(anchor="w")
    subtitle = label(intro, "Explore a plant image, compare its best matches and inspect related training examples.",
                     13, MUTED)
    subtitle.pack(anchor="w", pady=(7, 0))
    footer = tk.Frame(outer, bg=BG)
    footer.pack(side="bottom", fill="x", pady=(12, 0))
    dot = tk.Canvas(footer, width=12, height=16, bg=BG, highlightthickness=0)
    dot.pack(side="left", padx=(0, 5))
    dot.create_oval(3, 5, 8, 10, fill="#91a98b", outline="")
    app.status = label(footer, "Preparing your workspace...", 11, MUTED)
    app.status.pack(side="left", fill="x", expand=True)
    app.progress = ttk.Progressbar(footer, mode="indeterminate", length=80, style="Leaf.Horizontal.TProgressbar")
    app.progress.pack(side="right", padx=(12, 0))
    label(footer, "DEVELOPMENT PREVIEW", 9, MUTED, bold=True).pack(side="right")
    content = tk.Frame(outer, bg=BG)
    content.pack(fill="both", expand=True)
    content.columnconfigure(0, minsize=300, weight=0)
    content.columnconfigure(1, weight=1)
    content.rowconfigure(0, weight=1)
    query_card = Card(content)
    query_card.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
    left = query_card.body
    actions = tk.Frame(left, bg=WHITE)
    actions.pack(side="bottom", fill="x", pady=(12, 0))
    app.identify_button = Button(actions, "Analyze leaf  \u2192", app.identify, primary=True, state="disabled", height=44)
    app.identify_button.pack(fill="x", pady=(0, 7))
    app.export_button = Button(actions, "Export result", app.export, state="disabled", height=34)
    app.export_button.pack(fill="x")
    stack_area = tk.Frame(left, bg=WHITE)
    stack_area.pack(fill="both", expand=True)
    app.sidebar_canvas = tk.Canvas(stack_area, bg=WHITE, width=240, highlightthickness=0)
    stack_scroll = ttk.Scrollbar(stack_area, orient="vertical", command=app.sidebar_canvas.yview,
                                 style="Leaf.Vertical.TScrollbar")
    app.sidebar_canvas.configure(yscrollcommand=stack_scroll.set)
    stack_scroll.pack(side="right", fill="y", padx=(4, 0))
    app.sidebar_canvas.pack(side="left", fill="both", expand=True)
    left = tk.Frame(app.sidebar_canvas, bg=WHITE)
    stack_window = app.sidebar_canvas.create_window((0, 0), window=left, anchor="nw")
    left.bind("<Configure>", lambda _e: app.sidebar_canvas.configure(scrollregion=app.sidebar_canvas.bbox("all")))
    app.sidebar_canvas.bind("<Configure>", lambda e: app.sidebar_canvas.itemconfigure(stack_window, width=e.width))
    section(left, "01 / INPUT", "Your specimen")
    app.image_panel = tk.Canvas(left, bg="#f1f4ec", width=240, height=180, highlightthickness=0)
    app.image_panel.pack(fill="x", pady=(16, 10))
    app.image_panel.bind("<Configure>", lambda _e: app._draw_query())
    app.file_label = label(left, "No image selected", 11, MUTED, anchor="w", wraplength=252)
    app.file_label.pack(fill="x", pady=(0, 12))
    app.open_button = Button(left, "Choose an image", app.open_image, height=38)
    app.open_button.pack(fill="x")
    tk.Frame(left, bg=LINE, height=1).pack(fill="x", pady=18)
    label(left, "RECOGNITION MODEL", 10, MUTED, bold=True).pack(anchor="w")
    app.model_status = label(left, "Not connected", 13, INK, anchor="w", wraplength=252)
    app.model_status.pack(fill="x", pady=(8, 10))
    app.configure_button = Button(left, "Connect a model", app.configure_model, height=36)
    app.configure_button.pack(fill="x")
    label(left, "Development candidates only.\nWP2 / WP3 connections coming next.",
          10, MUTED, anchor="w", justify="left").pack(fill="x", pady=(10, 0))
    right = tk.Frame(content, bg=BG)
    right.grid(row=0, column=1, sticky="nsew")
    right.rowconfigure(1, weight=1)
    right.columnconfigure(0, weight=1)
    results = Card(right, height=360)
    results.grid(row=0, column=0, sticky="ew", pady=(0, 16))
    section(results.body, "02 / DISCOVER", "Best matches", "TOP 5")
    app.result_note = label(results.body, "Your ranked crop and disease matches will appear here.", 11, MUTED, anchor="w")
    app.result_note.pack(fill="x", pady=(8, 10))
    label(results.body, "Ranking scores compare matches; they are not confidence percentages.",
          10, MUTED, anchor="w").pack(side="bottom", fill="x", pady=(6, 0))
    app.table = RankedList(results.body)
    app.table.pack(fill="both", expand=True)
    app.table.bind("<<TreeviewSelect>>", app.select_prediction)
    reference_card = Card(right, height=290)
    reference_card.grid(row=1, column=0, sticky="nsew")
    references = reference_card.body
    section(references, "03 / EXPLORE", "Training references", "FITTING SET ONLY")
    app.reference_note = label(references, "Choose a match above or explore a labelled class.", 11, MUTED,
                               anchor="w", justify="left", wraplength=600)
    app.reference_note.pack(fill="x", pady=(8, 12))
    browse = tk.Frame(references, bg=WHITE)
    browse.pack(fill="x", pady=(0, 12))
    app.class_choice = ttk.Combobox(browse, state="disabled", width=28, style="Leaf.TCombobox")
    app.class_choice.pack(side="left", fill="x", expand=True, ipady=6)
    app.browse_button = Button(browse, "Browse class", app.browse_class, state="disabled", height=34)
    app.browse_button.configure(width=118)
    app.browse_button.pack(side="right", padx=(10, 0))
    gallery_area = tk.Frame(references, bg=WHITE)
    gallery_area.pack(fill="both", expand=True)
    app.gallery_canvas = tk.Canvas(gallery_area, bg=WHITE, highlightthickness=0, height=160)
    scrollbar = ttk.Scrollbar(gallery_area, orient="vertical", command=app.gallery_canvas.yview,
                              style="Leaf.Vertical.TScrollbar")
    app.gallery_canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y", padx=(8, 0))
    app.gallery_canvas.pack(side="left", fill="both", expand=True)
    app.gallery = tk.Frame(app.gallery_canvas, bg=WHITE)
    window = app.gallery_canvas.create_window((0, 0), window=app.gallery, anchor="nw")
    app.gallery.bind("<Configure>", lambda _e: app.gallery_canvas.configure(scrollregion=app.gallery_canvas.bbox("all")))
    app.gallery_canvas.bind("<Configure>", lambda e: app.gallery_canvas.itemconfigure(window, width=e.width))
    app.gallery_canvas.bind("<MouseWheel>", lambda e: app.gallery_canvas.yview_scroll(-int(e.delta/120), "units"))
    def scroll(event):
        widget = app.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget in (app.sidebar_canvas, app.gallery_canvas):
                widget.yview_scroll(-int(event.delta/120), "units")
                return "break"
            parent = widget.winfo_parent()
            widget = app.root.nametowidget(parent) if parent else None
    app.root.bind("<MouseWheel>", scroll)
    for column in range(2):
        app.gallery.columnconfigure(column, weight=1)
    app._empty_gallery()
    right.bind("<Configure>", lambda e: (
        app.result_note.configure(wraplength=max(250, e.width-42)),
        app.reference_note.configure(wraplength=max(250, e.width-42))))
    app._compact_layout = None
    def responsive(event):
        if event.widget != app.root:
            return
        compact = event.height < 860
        if compact == app._compact_layout:
            return
        app._compact_layout = compact
        results.configure(height=338 if compact else 360)
        app.table.set_compact(compact)
        header.pack_configure(pady=(0, 10 if compact else 20))
        intro.pack_configure(pady=(0, 12 if compact else 20))
        if compact:
            subtitle.pack_forget()
        else:
            subtitle.pack(anchor="w", pady=(7, 0))
    app.root.bind("<Configure>", responsive, add="+")
