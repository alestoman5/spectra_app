#!/usr/bin/env python3
"""IR Spectrum Analyzer – graphical user interface (Tkinter + matplotlib).

Run:  python app.py [obrazek.jpg]
"""
from __future__ import annotations

import csv
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from spectra import digitize as dg  # noqa: E402
from spectra.interpret import report_text  # noqa: E402
from spectra.pipeline import Result, Settings, analyse_image, analyse_xy, load_xy  # noqa: E402

APP_TITLE = "IR Spectrum Analyzer – interpretace IČ spekter"
IMAGE_TYPES = [("Obrázky", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("Vše", "*.*")]
DATA_TYPES = [("Data spektra", "*.csv *.txt *.dat *.xy *.dpt"), ("Vše", "*.*")]
STATUS_COLORS = {"potvrzeno": "#1b7f3b", "možné": "#b8860b", "vyloučeno": "#999999",
                 "nepřítomno": "#999999"}


class App(tk.Tk):
    def __init__(self, path: str | None = None):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1400x900")
        self.minsize(1000, 700)

        self.rgb: np.ndarray | None = None
        self.xy: tuple[np.ndarray, np.ndarray] | None = None
        self.source_path: str | None = None
        self.area: dg.PlotArea | None = None
        self.calibration: dg.XCalibration | None = None
        self.result: Result | None = None
        self.click_mode: str | None = None      # "corners" / "calib"
        self.clicks: list[tuple[float, float]] = []

        self._build_ui()
        if path:
            self.after(100, lambda: self.open_path(path))

    # ------------------------------------------------------------------ UI layout
    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))

        bar = ttk.Frame(self, padding=(6, 4))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="📂 Otevřít obrázek…", command=self.open_image).pack(side=tk.LEFT)
        ttk.Button(bar, text="📄 Otevřít data (CSV)…", command=self.open_data).pack(side=tk.LEFT, padx=4)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="▶ Analyzovat", style="Accent.TButton",
                   command=self.analyse).pack(side=tk.LEFT)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="⌗ Označit rohy grafu", command=self.start_corners).pack(side=tk.LEFT)
        ttk.Button(bar, text="↔ Kalibrovat osu x (2 body)", command=self.start_calib).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="↺ Automaticky", command=self.reset_manual).pack(side=tk.LEFT)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="💾 Export…", command=self.export).pack(side=tk.LEFT)

        main = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main.pack(fill=tk.BOTH, expand=True)
        top = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        main.add(top, weight=3)

        # left: image + settings
        left = ttk.Frame(top)
        top.add(left, weight=2)
        self._build_settings(left)
        self.fig_img = Figure(figsize=(5, 4), dpi=100)
        self.ax_img = self.fig_img.add_axes([0, 0, 1, 1])
        self.ax_img.axis("off")
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=left)
        self.canvas_img.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_img.mpl_connect("button_press_event", self.on_image_click)

        # right: spectrum
        right = ttk.Frame(top)
        top.add(right, weight=3)
        self.fig_sp = Figure(figsize=(7, 4), dpi=100)
        self.ax_sp = self.fig_sp.add_subplot(111)
        self.canvas_sp = FigureCanvasTkAgg(self.fig_sp, master=right)
        NavigationToolbar2Tk(self.canvas_sp, right).update()
        self.canvas_sp.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_sp.mpl_connect("button_press_event", self.on_spectrum_click)
        self._placeholder()

        # bottom: results
        nb = ttk.Notebook(main)
        main.add(nb, weight=2)
        self.summary_var = tk.StringVar(value="Otevřete obrázek spektra (foto, sken, screenshot) "
                                              "nebo data a stiskněte Analyzovat.")
        f_sum = ttk.Frame(nb, padding=6)
        nb.add(f_sum, text="Výsledek")
        ttk.Label(f_sum, textvariable=self.summary_var, wraplength=1300, justify=tk.LEFT,
                  font=("TkDefaultFont", 11, "bold")).pack(anchor=tk.W, fill=tk.X)
        self.groups_tree = self._tree(f_sum, ("skupina", "vzorec", "stav", "shoda", "pásy", "pozn"),
                                      ("Funkční skupina", "Fragment", "Stav", "Shoda",
                                       "Nalezené pásy (cm⁻¹)", "Poznámka"),
                                      (220, 110, 90, 60, 220, 500))
        f_peaks = ttk.Frame(nb, padding=6)
        nb.add(f_peaks, text="Pásy")
        self.peaks_tree = self._tree(f_peaks, ("wn", "int", "fwhm", "ass"),
                                     ("Vlnočet (cm⁻¹)", "Intenzita", "Šířka (cm⁻¹)", "Přiřazení"),
                                     (110, 80, 90, 900))
        self.peaks_tree.bind("<<TreeviewSelect>>", self.on_peak_select)
        f_rep = ttk.Frame(nb, padding=6)
        nb.add(f_rep, text="Zpráva")
        self.report = tk.Text(f_rep, wrap=tk.WORD, font=("TkFixedFont", 10))
        sb = ttk.Scrollbar(f_rep, command=self.report.yview)
        self.report.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.report.pack(fill=tk.BOTH, expand=True)

        self.status = tk.StringVar(value="Připraveno.")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W,
                  padding=(6, 2)).pack(side=tk.BOTTOM, fill=tk.X)

    def _tree(self, parent, cols, heads, widths):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        for c, h, w in zip(cols, heads, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor=tk.W, stretch=(c == cols[-1]))
        sb = ttk.Scrollbar(frame, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        for st, col in STATUS_COLORS.items():
            tree.tag_configure(st, foreground=col)
        return tree

    def _build_settings(self, parent):
        box = ttk.LabelFrame(parent, text="Nastavení", padding=6)
        box.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=4)
        self.v_mode = tk.StringVar(value="auto")
        self.v_color = tk.StringVar(value="auto")
        self.v_left = tk.DoubleVar(value=4000)
        self.v_step = tk.DoubleVar(value=500)
        self.v_use_right = tk.BooleanVar(value=False)
        self.v_right = tk.DoubleVar(value=400)
        self.v_smooth = tk.IntVar(value=9)
        self.v_prom = tk.DoubleVar(value=3.0)
        self.v_base = tk.BooleanVar(value=True)

        r = 0
        ttk.Label(box, text="Osa y:").grid(row=r, column=0, sticky=tk.W)
        ttk.Combobox(box, textvariable=self.v_mode, width=16, state="readonly",
                     values=("auto", "T", "A")).grid(row=r, column=1, sticky=tk.W)
        ttk.Label(box, text="(T = %transmitance, A = absorbance)").grid(row=r, column=2, columnspan=2,
                                                                         sticky=tk.W)
        r += 1
        ttk.Label(box, text="Barva křivky:").grid(row=r, column=0, sticky=tk.W)
        ttk.Combobox(box, textvariable=self.v_color, width=16, state="readonly",
                     values=("auto", "dark", "color", "blue")).grid(row=r, column=1, sticky=tk.W)
        ttk.Label(box, text="(dark = černá/šedá, color = červená ap., blue = modrá)").grid(row=r, column=2, columnspan=2,
                                                                          sticky=tk.W)
        r += 1
        ttk.Label(box, text="Levý konec osy x:").grid(row=r, column=0, sticky=tk.W)
        ttk.Entry(box, textvariable=self.v_left, width=8).grid(row=r, column=1, sticky=tk.W)
        ttk.Label(box, text="Krok popisků:").grid(row=r, column=2, sticky=tk.E)
        ttk.Entry(box, textvariable=self.v_step, width=8).grid(row=r, column=3, sticky=tk.W)
        r += 1
        ttk.Checkbutton(box, text="Pravý konec osy x =", variable=self.v_use_right).grid(
            row=r, column=0, sticky=tk.W)
        ttk.Entry(box, textvariable=self.v_right, width=8).grid(row=r, column=1, sticky=tk.W)
        ttk.Label(box, text="(jinak se kalibruje z popisků osy)").grid(row=r, column=2, columnspan=2,
                                                                        sticky=tk.W)
        r += 1
        ttk.Label(box, text="Vyhlazení (body):").grid(row=r, column=0, sticky=tk.W)
        ttk.Spinbox(box, from_=0, to=51, increment=2, textvariable=self.v_smooth, width=6).grid(
            row=r, column=1, sticky=tk.W)
        ttk.Checkbutton(box, text="Korekce základní linie", variable=self.v_base).grid(
            row=r, column=2, columnspan=2, sticky=tk.W)
        r += 1
        ttk.Label(box, text="Práh pásů (%):").grid(row=r, column=0, sticky=tk.W)
        sc = ttk.Scale(box, from_=0.5, to=15, variable=self.v_prom, orient=tk.HORIZONTAL)
        sc.grid(row=r, column=1, columnspan=2, sticky=tk.EW)
        self.prom_lbl = ttk.Label(box, text="3.0")
        self.prom_lbl.grid(row=r, column=3, sticky=tk.W)
        sc.configure(command=lambda v: self.prom_lbl.configure(text=f"{float(v):.1f}"))
        box.columnconfigure(2, weight=1)

    def _placeholder(self):
        self.ax_sp.clear()
        self.ax_sp.text(0.5, 0.5, "Zde se zobrazí digitalizované spektrum", ha="center",
                        va="center", transform=self.ax_sp.transAxes, color="#888")
        self.ax_sp.set_xticks([])
        self.ax_sp.set_yticks([])
        self.canvas_sp.draw_idle()

    # ------------------------------------------------------------------ input
    def open_image(self):
        p = filedialog.askopenfilename(title="Obrázek spektra", filetypes=IMAGE_TYPES)
        if p:
            self.open_path(p)

    def open_data(self):
        p = filedialog.askopenfilename(title="Data spektra", filetypes=DATA_TYPES)
        if p:
            self.open_path(p)

    def open_path(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".csv", ".txt", ".dat", ".xy", ".dpt"):
                self.xy = load_xy(path)
                self.rgb = None
            else:
                self.rgb = dg.load_image(path)
                self.xy = None
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Chyba", str(e))
            return
        self.source_path = path
        self.area = None
        self.calibration = None
        self.result = None
        self.title(f"{APP_TITLE} – {os.path.basename(path)}")
        if self.rgb is not None:
            self.area = dg.detect_plot_area(self.rgb)
            msg = ("Oblast grafu nalezena automaticky." if self.area.auto else
                   "Oblast grafu se nepodařilo najít – označte rohy ručně.")
            self.status.set(msg + " Zkontrolujte zelený rámeček a stiskněte Analyzovat.")
        else:
            self.status.set(f"Načteno {len(self.xy[0])} bodů. Stiskněte Analyzovat.")
        self.draw_image()
        self.analyse()

    def settings(self) -> Settings:
        def num(var, default):
            try:
                return float(var.get())
            except (tk.TclError, ValueError):
                return default
        return Settings(
            y_mode=self.v_mode.get(), color=self.v_color.get(),
            left_value=num(self.v_left, 4000), tick_step=num(self.v_step, 500),
            right_value=num(self.v_right, 400) if self.v_use_right.get() else None,
            smooth_window=int(num(self.v_smooth, 9)), prominence=num(self.v_prom, 3) / 100.0,
            baseline=bool(self.v_base.get()))

    # ------------------------------------------------------------------ analysis
    def analyse(self):
        if self.rgb is None and self.xy is None:
            messagebox.showinfo("Analýza", "Nejprve otevřete obrázek nebo data spektra.")
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            st = self.settings()
            if self.rgb is not None:
                self.result = analyse_image(self.rgb, st, self.area, self.calibration)
            else:
                self.result = analyse_xy(self.xy[0], self.xy[1], st, "data ze souboru")
        except Exception as e:  # noqa: BLE001
            self.config(cursor="")
            messagebox.showerror("Analýza selhala", f"{e}\n\nZkuste ručně označit rohy grafu, "
                                                     "změnit barvu křivky nebo kalibraci osy x.")
            return
        self.config(cursor="")
        self.status.set(self.result.info)
        self.draw_image()
        self.draw_spectrum()
        self.fill_results()

    # ------------------------------------------------------------------ drawing
    def draw_image(self):
        ax = self.ax_img
        ax.clear()
        ax.axis("off")
        if self.rgb is None:
            ax.text(0.5, 0.5, "Data načtena ze souboru\n(bez obrázku)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            self.canvas_img.draw_idle()
            return
        ax.imshow(self.rgb)
        if self.area is not None:
            c = np.vstack([self.area.corners, self.area.corners[:1]])
            ax.plot(c[:, 0], c[:, 1], "-", color="#00c853", lw=2)
            ax.plot(*self.area.corners[0], "o", color="#00c853")
        res = self.result
        if res is not None and res.trace is not None and self.area is not None:
            # project the traced curve back to the image
            w, h = self.area.size()
            hm = np.linalg.inv(self.area.homography(w, h))
            y = res.trace.bottom if res.y_mode == "T" else res.trace.top
            pts = np.stack([res.trace.px, y], axis=1).astype(np.float32).reshape(-1, 1, 2)
            import cv2
            img_pts = cv2.perspectiveTransform(pts, hm).reshape(-1, 2)
            ax.plot(img_pts[:, 0], img_pts[:, 1], "-", color="#2962ff", lw=1, alpha=0.8)
        for x, y in self.clicks:
            ax.plot(x, y, "x", color="#ff1744", ms=10, mew=2)
        if self.click_mode:
            txt = {"corners": "Klikněte na 4 rohy oblasti grafu (libovolné pořadí)",
                   "calib": "Klikněte na 2 popsané rysky osy x"}[self.click_mode]
            ax.set_title(txt, color="#d50000", fontsize=10)
        self.canvas_img.draw_idle()

    def draw_spectrum(self, highlight: float | None = None):
        res = self.result
        ax = self.ax_sp
        ax.clear()
        if res is None:
            self._placeholder()
            return
        sp = res.processed
        ax.plot(sp.x, sp.y, color="#1f3b73", lw=1)
        interp = res.interpretation
        for p in res.peaks:
            assigned = bool(interp.assignments.get(p.wavenumber)) and not p.artefact
            col = "#c62828" if assigned else ("#9e9e9e" if p.artefact else "#607d8b")
            ax.plot(p.wavenumber, p.height, "v", color=col, ms=5)
            if p.height > 0.06 or assigned:
                ax.annotate(f"{p.wavenumber:.0f}", (p.wavenumber, p.height),
                            xytext=(0, 4), textcoords="offset points", rotation=90,
                            ha="center", va="bottom", fontsize=7, color=col)
        if highlight is not None:
            ax.axvline(highlight, color="#ff9800", lw=1.5, alpha=0.8)
        ax.set_xlim(max(sp.x), min(sp.x))
        ax.set_ylim(-0.03, 1.22)
        ax.set_xlabel("Vlnočet (cm⁻¹)")
        ax.set_ylabel("Absorbance (normovaná)")
        ax.set_title(os.path.basename(self.source_path or ""), fontsize=10)
        ax.grid(alpha=0.25)
        self.fig_sp.tight_layout()
        self.canvas_sp.draw_idle()

    def fill_results(self):
        res = self.result
        interp = res.interpretation
        self.summary_var.set(interp.summary)
        for t in (self.groups_tree, self.peaks_tree):
            t.delete(*t.get_children())
        for r in interp.groups:
            if r.status in ("potvrzeno", "možné") or r.status == "vyloučeno":
                bands = ", ".join(f"{m.peak.wavenumber:.0f}" for m in r.matches if m.peak)
                note = r.reason or r.group.note
                self.groups_tree.insert("", tk.END, values=(
                    r.group.name, r.group.formula, r.status, f"{r.score * 100:.0f} %", bands, note),
                    tags=(r.status,))
        for p in res.peaks:
            ass = "; ".join(interp.assignments.get(p.wavenumber, [])) or "–"
            self.peaks_tree.insert("", tk.END, iid=str(p.wavenumber), values=(
                f"{p.wavenumber:.0f}", p.intensity + (", br" if p.broad else ""),
                f"{p.width:.0f}", ass))
        self.report.delete("1.0", tk.END)
        self.report.insert(tk.END, (res.info + "\n\n" if res.info else "") +
                           report_text(interp))

    # ------------------------------------------------------------------ interaction
    def start_corners(self):
        if self.rgb is None:
            return
        self.click_mode, self.clicks = "corners", []
        self.status.set("Klikněte do obrázku na 4 rohy oblasti grafu (průsečíky os/rámečku).")
        self.draw_image()

    def start_calib(self):
        if self.rgb is None:
            return
        self.click_mode, self.clicks = "calib", []
        self.status.set("Klikněte na dvě rysky osy x se známým vlnočtem (např. 4000 a 1000).")
        self.draw_image()

    def reset_manual(self):
        if self.rgb is None:
            return
        self.area = dg.detect_plot_area(self.rgb)
        self.calibration = None
        self.click_mode, self.clicks = None, []
        self.analyse()

    def on_image_click(self, ev):
        if not self.click_mode or ev.xdata is None or ev.button != 1:
            return
        self.clicks.append((ev.xdata, ev.ydata))
        if self.click_mode == "corners" and len(self.clicks) == 4:
            pts = np.array(self.clicks, np.float32)
            s = pts.sum(1)
            d = np.diff(pts, axis=1).ravel()
            tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
            tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
            self.area = dg.PlotArea(np.float32([tl, tr, br, bl]), auto=False)
            self.calibration = None
            self.click_mode, self.clicks = None, []
            self.analyse()
            return
        if self.click_mode == "calib" and len(self.clicks) == 2:
            vals = []
            for i, (x, y) in enumerate(self.clicks):
                v = simpledialog.askfloat("Kalibrace osy x", f"Vlnočet {i + 1}. bodu (cm⁻¹):",
                                          parent=self)
                if v is None:
                    self.click_mode, self.clicks = None, []
                    self.draw_image()
                    return
                vals.append(v)
            if self.area is None:
                self.area = dg.detect_plot_area(self.rgb)
            px = dg.map_points(np.array(self.clicks), self.area)[:, 0]
            if abs(px[1] - px[0]) < 5 or vals[0] == vals[1]:
                messagebox.showerror("Kalibrace", "Body musí být od sebe vzdálené a mít různé hodnoty.")
            else:
                self.calibration = dg.XCalibration.from_points(px[0], vals[0], px[1], vals[1])
            self.click_mode, self.clicks = None, []
            self.analyse()
            return
        self.draw_image()

    def on_spectrum_click(self, ev):
        if self.result is None or ev.xdata is None or ev.inaxes is not self.ax_sp:
            return
        if self.canvas_sp.toolbar is not None and self.canvas_sp.toolbar.mode:
            return
        peaks = self.result.peaks
        if not peaks:
            return
        p = min(peaks, key=lambda p: abs(p.wavenumber - ev.xdata))
        iid = str(p.wavenumber)
        if self.peaks_tree.exists(iid):
            self.peaks_tree.selection_set(iid)
            self.peaks_tree.see(iid)
        ass = "; ".join(self.result.interpretation.assignments.get(p.wavenumber, [])) or "nepřiřazeno"
        self.status.set(f"Pás {p.wavenumber:.0f} cm⁻¹ ({p.intensity}"
                        f"{', široký' if p.broad else ''}): {ass}")

    def on_peak_select(self, _ev=None):
        sel = self.peaks_tree.selection()
        if sel and self.result is not None:
            self.draw_spectrum(highlight=float(sel[0]))

    # ------------------------------------------------------------------ export
    def export(self):
        if self.result is None:
            messagebox.showinfo("Export", "Nejprve proveďte analýzu.")
            return
        base = os.path.splitext(os.path.basename(self.source_path or "spektrum"))[0]
        d = filedialog.askdirectory(title="Složka pro export")
        if not d:
            return
        res = self.result
        stem = os.path.join(d, base)
        with open(stem + "_pasy.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["vlnocet_cm-1", "intenzita", "siroky", "vyska_norm", "FWHM_cm-1", "prirazeni"])
            for p in res.peaks:
                w.writerow([f"{p.wavenumber:.1f}", p.intensity, "ano" if p.broad else "ne",
                            f"{p.height:.3f}", f"{p.width:.0f}",
                            "; ".join(res.interpretation.assignments.get(p.wavenumber, []))])
        with open(stem + "_spektrum.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["vlnocet_cm-1", "absorbance_surova", "absorbance_zpracovana"])
            for x, a, b in zip(res.raw.x, res.raw.y, res.processed.y):
                w.writerow([f"{x:.1f}", f"{a:.5f}", f"{b:.5f}"])
        with open(stem + "_zprava.txt", "w", encoding="utf-8") as f:
            f.write(res.info + "\n\n" + report_text(res.interpretation))
        self.fig_sp.savefig(stem + "_spektrum.png", dpi=200)
        self.status.set(f"Exportováno do {d}: {base}_pasy.csv, _spektrum.csv, _zprava.txt, _spektrum.png")


def main():
    app = App(sys.argv[1] if len(sys.argv) > 1 else None)
    app.mainloop()


if __name__ == "__main__":
    main()
