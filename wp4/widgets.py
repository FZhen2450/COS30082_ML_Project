"""Small native Tk widgets for the LeafLens visual language."""
import tkinter as tk

BG = "#f5f6f2"
WHITE = "#ffffff"
INK = "#21372d"
MUTED = "#7b877f"
GREEN = "#246348"
LINE = "#e5eae3"
FONT = "Segoe UI"


def rounded(canvas, box, radius=16, **kwargs):
    x, y, right, bottom = box
    r = min(radius, (right - x) / 2, (bottom - y) / 2)
    return canvas.create_polygon(
        x+r, y, right-r, y, right, y, right, y+r,
        right, bottom-r, right, bottom, right-r, bottom,
        x+r, bottom, x, bottom, x, bottom-r, x, y+r, x, y,
        smooth=True, splinesteps=24, **kwargs)


def label(parent, text, size=13, color=INK, bold=False, **kwargs):
    return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color,
                    font=(FONT, -size, "bold" if bold else "normal"), **kwargs)


class Card(tk.Canvas):
    def __init__(self, parent, height=300, padding=20, fill=WHITE):
        super().__init__(parent, bg=parent.cget("bg"), height=height, width=300,
                         highlightthickness=0, bd=0)
        self.fill, self.padding = fill, padding
        self.body = tk.Frame(self, bg=fill)
        self.window = self.create_window(padding, padding, window=self.body, anchor="nw")
        self.bind("<Configure>", self._resize)

    def _resize(self, event):
        self.delete("surface")
        rounded(self, (1, 1, event.width-1, event.height-1), 20,
                fill=self.fill, outline=LINE, width=1, tags="surface")
        self.tag_lower("surface")
        self.itemconfigure(self.window, width=max(1, event.width-self.padding*2),
                           height=max(1, event.height-self.padding*2))


class Button(tk.Canvas):
    def __init__(self, parent, text, command, primary=False, state="normal", height=42):
        super().__init__(parent, height=height, width=160, bg=parent.cget("bg"),
                         highlightthickness=0, bd=0, takefocus=True)
        self.text, self.command, self.primary = text, command, primary
        self._state, self.hover = state, False
        self.bind("<Configure>", lambda _e: self._draw())
        self.bind("<Enter>", lambda _e: self._hover(True))
        self.bind("<Leave>", lambda _e: self._hover(False))
        self.bind("<Button-1>", lambda _e: self.invoke())
        self.bind("<Return>", lambda _e: self.invoke())
        self.bind("<space>", lambda _e: self.invoke())
        self.bind("<FocusIn>", lambda _e: self._draw())
        self.bind("<FocusOut>", lambda _e: self._draw())

    def _hover(self, value):
        self.hover = value
        self._draw()

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs.update(cnf)
        if "state" in kwargs:
            self._state = kwargs.pop("state")
        if "text" in kwargs:
            self.text = kwargs.pop("text")
        if kwargs:
            super().configure(**kwargs)
        self._draw()

    config = configure

    def __getitem__(self, key):
        return self._state if key == "state" else super().__getitem__(key)

    def invoke(self):
        if self._state != "disabled":
            self.focus_set()
            return self.command()

    def _draw(self):
        if not hasattr(self, "text"):
            return
        self.delete("all")
        disabled = self._state == "disabled"
        if self.primary:
            fill = "#dce5dc" if disabled else "#194f38" if self.hover else GREEN
            text = "#8c9e90" if disabled else WHITE
            border = fill
        else:
            fill = "#f5f7f3" if disabled else "#edf3ec" if self.hover else WHITE
            text, border = ("#a2aca4" if disabled else INK), LINE
        if self.focus_get() == self and not disabled:
            border = GREEN
        rounded(self, (1, 1, self.winfo_width()-1, self.winfo_height()-1), 12,
                fill=fill, outline=border, width=1.5)
        self.create_text(self.winfo_width()/2, self.winfo_height()/2,
                         text=self.text, fill=text, font=(FONT, -13, "bold"))
        super().configure(cursor="arrow" if disabled else "hand2")


class RankedList(tk.Frame):
    """Selectable ranking cards with the minimal Treeview selection interface."""
    def __init__(self, parent):
        super().__init__(parent, bg=WHITE)
        self.rows = {}
        self.selected = ()
        self.compact = False
        self.enabled = True
        self._empty()

    def _empty(self):
        self.empty = tk.Frame(self, bg=WHITE)
        self.empty.pack(fill="both", expand=True)
        icon = tk.Canvas(self.empty, width=90, height=74, bg=WHITE, highlightthickness=0)
        icon.pack(pady=(26, 12))
        rounded(icon, (18, 9, 74, 65), 14, fill="#f0f5ed", outline="")
        icon.create_polygon(35, 49, 29, 28, 57, 22, 62, 38, 35, 49,
                            smooth=True, fill="#a7c3a4", outline="")
        icon.create_line(33, 49, 53, 29, fill=GREEN, width=1.5)
        label(self.empty, "A closer look starts here", 17, bold=True).pack()
        label(self.empty, "Add a leaf image and connect a model to see the best matches.",
              12, MUTED, wraplength=450).pack(pady=8)

    def get_children(self):
        return tuple(self.rows)

    def delete(self, *items):
        for iid in items:
            self.rows.pop(iid)["canvas"].destroy()
        self.selected = ()
        if not self.rows:
            if hasattr(self, "empty"):
                self.empty.destroy()
            self._empty()

    def insert(self, _parent, _where, iid, values):
        if not self.rows:
            self.empty.destroy()
        canvas = tk.Canvas(self, height=(44 if not self.rows else 32) if self.compact else (50 if not self.rows else 36), bg=WHITE,
                           highlightthickness=0, cursor="hand2", takefocus=True)
        canvas.pack(fill="x", pady=(0, 3))
        self.rows[iid] = {"canvas": canvas, "values": values}
        canvas.bind("<Configure>", lambda _e, key=iid: self._draw(key))
        canvas.bind("<Button-1>", lambda _e, key=iid: self._choose(key))
        canvas.bind("<Return>", lambda _e, key=iid: self._choose(key))
        canvas.bind("<space>", lambda _e, key=iid: self._choose(key))
        canvas.bind("<Up>", lambda _e, key=iid: self._move(key, -1))
        canvas.bind("<Down>", lambda _e, key=iid: self._move(key, 1))

    def _move(self, key, delta):
        ids = list(self.rows)
        self._choose(ids[max(0, min(len(ids)-1, ids.index(key)+delta))])

    def _choose(self, iid):
        if self.enabled:
            self.selection_set(iid)

    def selection(self):
        return self.selected

    def selection_set(self, iid):
        if iid not in self.rows:
            return
        self.selected = (iid,)
        self.rows[iid]["canvas"].focus_set()
        for key in self.rows:
            self._draw(key)
        self.event_generate("<<TreeviewSelect>>", when="tail")

    def set_compact(self, compact):
        self.compact = compact
        for row in self.rows.values():
            first = row["values"][0] == 1
            row["canvas"].configure(height=(44 if first else 32) if compact else (50 if first else 36))

    def _draw(self, iid):
        row = self.rows[iid]
        canvas = row["canvas"]
        rank, crop, disease, score = row["values"]
        first, selected = rank == 1, self.selected == (iid,)
        w, h = canvas.winfo_width(), canvas.winfo_height()
        canvas.delete("all")
        fill = "#eaf3e8" if first else "#f7faf5" if selected else "#fafbf9"
        rounded(canvas, (1, 1, w-1, h-1), 12, fill=fill,
                outline="#91b38e" if selected else fill, width=1)
        canvas.create_oval(13, h/2-13, 39, h/2+13, fill=GREEN if first else "#edf0e9", outline="")
        canvas.create_text(26, h/2, text=f"{rank:02}", fill=WHITE if first else MUTED,
                           font=(FONT, -11, "bold"))
        max_chars = max(12, int((w-176)/7))
        name = disease.replace("_", " ")
        if len(name) > max_chars:
            name = name[:max_chars-1] + "…"
        canvas.create_text(52, h/2-8, text=name.capitalize(), anchor="w", fill=INK,
                           font=(FONT, -14 if first else -13, "bold"))
        canvas.create_text(52, h/2+11, text=crop.replace("_", " ").capitalize(),
                           anchor="w", fill=MUTED, font=(FONT, -11))
        canvas.create_text(w-16, h/2-7, text=score, anchor="e", fill=GREEN if first else INK,
                           font=(FONT, -13, "bold"))
        canvas.create_text(w-16, h/2+10, text="RANK SCORE", anchor="e", fill=MUTED,
                           font=(FONT, -8))
