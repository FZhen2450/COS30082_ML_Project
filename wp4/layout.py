"""Three-column workspace: controls, image and rankings, permanent references."""
import tkinter as tk
from tkinter import ttk

from .model_picker import ModelPanel
from .widgets import BG, WHITE, INK, MUTED, GREEN, LINE, Button, Card, RankedList, label, rounded


def build(app):
    background = tk.Canvas(app.root, bg=BG, highlightthickness=0)
    background.pack(fill="both", expand=True)
    shell = tk.Frame(background, bg=BG)
    shell_window = background.create_window(18, 88, window=shell, anchor="nw")
    app.scope_caption = 'LOCAL WORKSPACE'
    app.update_scope_caption = lambda text: background.itemconfigure('scope-label', text=text)
    shell.rowconfigure(0, weight=1)
    shell.columnconfigure(0, minsize=280)
    shell.columnconfigure(1, weight=1, minsize=480)
    shell.columnconfigure(2, minsize=340)

    def paint(event):
        background.delete('decoration')
        for y in range(0, event.height, 8):
            ratio = y / max(1, event.height)
            start, end = (226, 238, 232), (244, 245, 236)
            color = '#' + ''.join(f'{round(a+(b-a)*ratio):02x}' for a, b in zip(start, end))
            background.create_rectangle(0, y, event.width, y+8, fill=color, outline='', tags='decoration')
        background.create_text(28, 32, text='LeafLens', anchor='w', fill=INK,
                               font=('Segoe UI', -30, 'bold'), tags='decoration')
        background.create_text(188, 35, text='A little insight. A healthier leaf.', anchor='w',
                               fill=MUTED, font=('Segoe UI', -17), tags='decoration')
        rounded(background, (event.width-238, 18, event.width-24, 57), 18,
                fill='#d9e9df', outline='', tags='decoration')
        background.create_oval(event.width-218, 33, event.width-210, 41,
                               fill='#b18b42', outline='', tags='decoration')
        background.create_text(event.width-115, 37, text=app.scope_caption, fill=GREEN,
                               font=('Segoe UI', -13, 'bold'), tags=('decoration', 'scope-label'))
        background.create_line(26, 71, event.width-26, 71, fill='#cddfd3', tags='decoration')
        background.itemconfigure(shell_window, width=max(1, event.width-36), height=max(1, event.height-132))
        background.tag_lower('decoration')

    background.bind('<Configure>', paint)
    sidebar_card = Card(shell, padding=18)
    sidebar_card.grid(row=0, column=0, sticky='nsew', padx=(0, 16))
    sidebar = sidebar_card.body
    actions = tk.Frame(sidebar, bg=WHITE)
    actions.pack(side='bottom', fill='x', pady=(12, 0))
    app.identify_button = Button(actions, 'Analyze leaf  \u2192', app.identify,
                                 primary=True, state='disabled', height=52)
    app.identify_button.pack(fill='x', pady=(0, 9))
    app.export_button = Button(actions, 'Export result', app.export, state='disabled', height=44)
    app.export_button.pack(fill='x')
    area = tk.Frame(sidebar, bg=WHITE)
    area.pack(fill='both', expand=True)
    app.sidebar_canvas = tk.Canvas(area, bg=WHITE, highlightthickness=0, width=240, yscrollincrement=20)
    side_scroll = ttk.Scrollbar(area, orient='vertical', command=app.sidebar_canvas.yview,
                                style='Leaf.Vertical.TScrollbar')
    app.sidebar_canvas.configure(yscrollcommand=side_scroll.set)
    side_scroll.pack(side='right', fill='y', padx=(4, 0))
    app.sidebar_canvas.pack(side='left', fill='both', expand=True)
    controls = tk.Frame(app.sidebar_canvas, bg=WHITE)
    side_window = app.sidebar_canvas.create_window((0, 0), window=controls, anchor='nw')
    controls.bind('<Configure>', lambda e: app.sidebar_canvas.configure(scrollregion=(0, 0, e.width, e.height)))
    app.sidebar_canvas.bind('<Configure>', lambda e: app.sidebar_canvas.itemconfigure(side_window, width=e.width))
    label(controls, 'YOUR WORKSPACE', 11, GREEN, bold=True).pack(anchor='w')
    label(controls, 'Start with a leaf', 20, bold=True).pack(anchor='w', pady=(6, 18))
    app.open_button = Button(controls, 'Choose an image', app.open_image, height=48)
    app.open_button.pack(fill='x')
    app.file_label = label(controls, 'No image selected', 12, MUTED, anchor='nw',
                           justify='left', wraplength=226, height=2)
    app.file_label.pack(fill='x', pady=(10, 0))
    tk.Frame(controls, bg=LINE, height=1).pack(fill='x', pady=16)
    label(controls, 'RECOGNITION MODEL', 11, GREEN, bold=True).pack(anchor='w', pady=(0, 12))
    app.model_panel = ModelPanel(controls, app.connect_model)
    app.model_panel.pack(fill='x')
    app.model_panel.set_enabled(False)
    app.configure_button = app.model_panel.load_button
    app.model_dialog = app.model_panel
    app.model_status = label(controls, 'No model loaded', 12, MUTED, anchor='nw',
                             justify='left', wraplength=226, height=3)
    app.model_status.pack(fill='x', pady=(12, 0))

    center = tk.Frame(shell, bg=BG)
    center.grid(row=0, column=1, sticky='nsew', padx=(0, 16))
    center.columnconfigure(0, weight=1)
    center.rowconfigure(0, weight=6, uniform='center')
    center.rowconfigure(1, weight=4, uniform='center')
    specimen = Card(center, padding=12, height=1)
    specimen.grid(row=0, column=0, sticky='nsew', pady=(0, 14))
    heading = tk.Frame(specimen.body, bg=WHITE)
    heading.pack(fill='x')
    label(heading, 'A closer look', 21, bold=True).pack(side='left')
    label(heading, 'FIT WHOLE IMAGE', 10, GREEN, bold=True).pack(side='right')
    app.image_panel = tk.Canvas(specimen.body, bg='#eaf2e9', highlightthickness=0, height=180)
    app.image_panel.pack(fill='both', expand=True, pady=(8, 0))
    app.image_panel.bind('<Configure>', lambda _e: app._draw_query())
    results = Card(center, padding=12, height=1)
    results.grid(row=1, column=0, sticky='nsew')
    heading = tk.Frame(results.body, bg=WHITE)
    heading.pack(fill='x')
    label(heading, 'Best matches', 21, bold=True).pack(side='left')
    label(heading, 'TOP 5', 11, GREEN, bold=True).pack(side='right')
    app.result_note = label(results.body, 'Analyze your image to see crop and disease matches.',
                            12, MUTED, anchor='w', justify='left', wraplength=550)
    app.result_note.pack(fill='x', pady=(3, 4))
    app.table = RankedList(results.body)
    app.table.set_compact(True)
    app.table.pack(fill='both', expand=True)
    app.table.bind('<<TreeviewSelect>>', app.select_prediction)
    score_note = label(results.body, 'Ranking scores, not confidence percentages.', 11, MUTED, anchor='w')
    score_note.pack(fill='x', pady=(3, 0))
    results.body.bind('<Configure>', lambda e: app.result_note.configure(wraplength=max(280, e.width-4)), add='+')

    reference_card = Card(shell, padding=18)
    reference_card.grid(row=0, column=2, sticky='nsew')
    references = reference_card.body
    app.workspace = references
    label(references, 'A LITTLE CONTEXT', 11, GREEN, bold=True).pack(anchor='w')
    label(references, 'Training references', 21, bold=True).pack(anchor='w', pady=(6, 0))
    app.reference_note = label(references, 'Choose a match or browse a class. Fitting images only.',
                               12, MUTED, anchor='w', justify='left', wraplength=290)
    app.reference_note.pack(fill='x', pady=(12, 14))
    app.class_choice = ttk.Combobox(references, state='disabled', width=20,
                                   style='Leaf.TCombobox', font=('Segoe UI', -16))
    app.class_choice.pack(fill='x', ipady=3)
    app.browse_button = Button(references, 'Browse class', app.browse_class, state='disabled', height=42)
    app.browse_button.pack(fill='x', pady=(8, 14))
    gallery_area = tk.Frame(references, bg=WHITE)
    gallery_area.pack(fill='both', expand=True)
    app.page_canvas = tk.Canvas(gallery_area, bg=WHITE, highlightthickness=0, width=290, yscrollincrement=24)
    scrollbar = ttk.Scrollbar(gallery_area, orient='vertical', command=app.page_canvas.yview,
                              style='Leaf.Vertical.TScrollbar')
    app.page_canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right', fill='y', padx=(5, 0))
    app.page_canvas.pack(side='left', fill='both', expand=True)
    app.gallery = tk.Frame(app.page_canvas, bg=WHITE)
    gallery_window = app.page_canvas.create_window((0, 0), window=app.gallery, anchor='nw')
    app.gallery.bind('<Configure>', lambda e: app.page_canvas.configure(scrollregion=(0, 0, e.width, e.height)))
    app.page_canvas.bind('<Configure>', lambda e: app.page_canvas.itemconfigure(gallery_window, width=e.width))
    app.gallery.columnconfigure(0, weight=1)
    app.gallery_columns = 1
    app._empty_gallery()
    app.status = label(background, 'Preparing your workspace...', 12, MUTED, anchor='w')
    app.progress = ttk.Progressbar(background, mode='indeterminate', length=100,
                                   style='Leaf.Horizontal.TProgressbar')
    app.status.place(x=26, rely=1, y=-32, relwidth=.75)
    app.progress.place(relx=1, x=-128, rely=1, y=-27, width=100)

    def scroll(event):
        widget = event.widget
        if widget.winfo_toplevel() != app.root:
            return
        target = None
        while widget is not None:
            if widget == references:
                target = app.page_canvas
                break
            if widget == sidebar:
                target = app.sidebar_canvas
                break
            widget = widget.master
        if target is None:
            return
        units = -3 if event.num == 4 else 3 if event.num == 5 else (
            -max(1, abs(event.delta)//120) * (1 if event.delta > 0 else -1))
        target.yview_scroll(units, 'units')
        return 'break'

    app.root.bind('<MouseWheel>', scroll)
    app.root.bind('<Button-4>', scroll)
    app.root.bind('<Button-5>', scroll)
