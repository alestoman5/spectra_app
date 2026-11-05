# IR Spectrum Analyzer

Desktop app (Python + Tkinter). You give it a **picture of an infrared spectrum**: a photo of
an OMNIC printout, a scan or a screenshot. The app digitises the curve, finds the absorption
bands and lists the **functional groups** the compound contains. It also suggests a
compound class and the closest match from a small library of reference compounds.

The interpretation follows the procedure taught in the CACH II presentation *Interpretace
infračervených spekter* (ÚACh VŠCHT Praha): the "Rozhodovací schema" decision tree, the
"rychlokuchařka" (quick cookbook) and the band tables on the per-class slides.

1. **Skeleton first.** Bands at 3100–3000 cm⁻¹ point to an arene or alkene; arene also needs
   the ring ν(C=C) bands, the overtone pattern or strong γ(CH). Bands at 3000–2850 cm⁻¹ mean
   alkane C–H.
2. **Strong band at 1800–1650 cm⁻¹ (C=O)?**
   - **Yes:** a broad O–H band at 3300–2500 that reaches below 3000 → **carboxylic acid**.
     Otherwise two strong C–O bands at 1310–1050 → **ester**. Otherwise a Fermi doublet at
     2830–2650 → **aldehyde**. Otherwise → **ketone**, by exclusion.
   - **No:** a broad band at 3550–3230 → **alcohol / phenol**, typed by the C–O position.
     N–H bands at 3550–3250 → **amine** (2 bands = primary, 1 = secondary). A sharp band at
     2260–2100 → **nitrile / alkyne**. Two strong bands at 1570–1320 → **nitro compound**.
     A strong C–O band without O–H → **ether**.
3. **Look up all bands of the class found** ("dohledat všechny pásy"). A missing
   characteristic band means the group is not present.
4. **Details:** CH3 singlet or doublet (branching), CH2 rocking (long chain), aromatic
   substitution (mono/ortho/meta/para), alcohol type, ester type, C=O position.
5. **End check:** report any strong band above 1300 cm⁻¹ that is still unexplained.

Every step is shown in the app, so you can follow and check the reasoning.

The user interface is in Czech.

![screenshot](docs/screenshot.png)

## Installation

```bash
pip install -r requirements.txt
```

Tkinter must be available. On Debian/Ubuntu install it with `sudo apt install python3-tk`;
on Windows and macOS it ships with the python.org installer.

## Usage

### GUI

```bash
python app.py                 # or: python app.py fotka_spektra.jpg
```

1. **Otevřít obrázek…** opens a photo, scan or screenshot of the spectrum. **Otevřít data
   (CSV)…** opens a two-column file of wavenumber and intensity.
2. The plot area is detected automatically and shown as a green frame. The traced curve is
   drawn in blue over the image.
3. **Analyzovat** shows the spectrum in normalised absorbance. Detected bands are marked:
   red for assigned bands, grey for unassigned ones.
4. The result tabs:
   - **Výsledek**: summary, the step-by-step decision path, and the confirmed, possible and
     excluded groups. Each group lists the bands found, the bands missing, and why it was
     excluded.
   - **Pásy**: every band with its intensity (s/m/w, br) and assignment. Clicking a band
     highlights it in the plot.
   - **Zpráva**: a text report you can copy into a lab protocol.
5. **Export…** saves the band table (CSV), the digitised spectrum (CSV), the report (TXT) and
   the plot (PNG).

When the automatic steps fail:

| Problem | Fix |
|---|---|
| Wrong frame | **Označit rohy grafu**, then click the 4 corners of the plot area |
| Wrong x axis | **Kalibrovat osu x (2 body)**, then click two labelled ticks and type their wavenumbers. You can also set *Levý konec osy x* and *Krok popisků*, or tick *Pravý konec osy x* |
| Several spectra stacked in one picture | **Označit rohy grafu** around one panel; if that panel has no axis labels, tick *Pravý konec osy x* and enter the range |
| Wrong y axis | set *Osa y* to `T` (%transmittance) or `A` (absorbance) |
| Curve not found | set *Barva křivky* (`dark`, `color` = red/green…, `blue`) |
| Too many or too few bands | move the *Práh pásů* slider |

### Command line

```bash
python analyze.py spektrum.jpg
```

```bash
python analyze.py spektrum.csv --mode T
```

### Examples

`examples/` holds synthetic example images (polystyrene, ethyl acetate, benzoic acid,
benzaldehyde). To regenerate them:

```bash
python examples/make_examples.py
```

## How it works

```
image ─► plot-area detection ─► perspective rectification ─► curve extraction ─► x calibration
      ─► resampling (1 cm⁻¹) ─► %T → absorbance ─► Savitzky–Golay smoothing ─► ALS baseline
      ─► peak detection (prominence, FWHM) ─► decision scheme (presentation) ─► band look-up
      ─► groups + report
```

| Module | Purpose |
|---|---|
| `spectra/digitize.py` | Finds the axes: long lines from a Hough transform, the L-shaped corner with tick marks, and the top/right frame. Corrects perspective with a homography. Removes frame lines, labels and blue handwritten notes, then traces the curve column by column with continuity tracking. Calibrates the x axis automatically from the centres of the tick labels. It handles both OMNIC printouts (4000, 3500, … starting at the axis) and NIST-style plots (3000, 2000, 1000) by choosing the label value and step that give a plausible mid-IR range. |
| `spectra/signal.py` | Resampling, %T → A conversion, Savitzky–Golay smoothing, asymmetric least-squares baseline, `scipy.signal.find_peaks`. Classifies intensity (s/m/w) and flags broad bands. Marks the CO₂ artefact at 2360/2335 cm⁻¹. |
| `spectra/ir_table.py` | Band tables from the presentation slides (alkanes … nitro compounds), plus amides and anhydrides outside the scheme. |
| `spectra/interpret.py` | The decision tree, the band look-up and the details; records every step, then writes the summary and report. |
| `spectra/references.py` | Approximate band lists of about 60 compounds, including the ones shown in the presentation. Only compounds of the class found by the tree are compared. |
| `tools/extract_presentation_spectra.py` | Cuts the labelled sample spectra out of the presentation PDF (and the photos out of the lab protocol) into the git-ignored `local_data/`. |
| `tools/evaluate.py` | Runs the whole pipeline on the labelled set and prints expected vs. predicted class. |
| `spectra/pipeline.py` | Joins the steps together (used by the GUI and the CLI). |

## Tests

```bash
python -m pytest
```

The tests build synthetic spectra from Lorentzian bands, render them as images with
matplotlib (%T and A, black, coloured and slightly rotated), then digitise and interpret
them. They check band positions to within ±10 cm⁻¹ and the expected compound class.

### Calibration on the presentation spectra

The sample spectra from the presentation are not in the repository. To create them
locally (needs `pdfimages` from poppler-utils):

```bash
python tools/extract_presentation_spectra.py Prezentace.pdf --protocol protokol_ir.pdf
```

```bash
python tools/evaluate.py
```

`tests/test_presentation.py` then runs on that set; without the data it is skipped. The
current result is 29/29 classes correct: 24 spectra from the presentation plus 5 photos
from the lab protocol.

## Limitations

- The digitised spectrum is only as accurate as the picture. Expect a few cm⁻¹ of error
  (about 5–10 cm⁻¹ on phone photos), and band intensities are approximate.
- The app identifies functional groups and the compound class. Naming the exact molecule
  needs a comparison with a spectral library; the built-in reference list only gives a
  hint.
- Weak shoulders on the slope of a strong band, such as 721 cm⁻¹ next to a steep edge, may
  be missed. Lower the band threshold to catch them.
- Only the classes in the presentation's scheme are decided automatically. Amides and
  anhydrides are recognised as a side branch; tertiary amides (e.g. DMF) come out as
  ketones. Halogen compounds and inorganic salts are not covered.
- Gas-phase spectra (NIST WebBook) have bands shifted by up to ~20–30 cm⁻¹ against
  liquid-phase tables, and their aromatic C–H bands can be very weak. The rules allow for
  this, but the skeleton of e.g. ethyl benzoate can come out as "neurčeno".
