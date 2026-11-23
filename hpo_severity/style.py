"""
Matplotlib settings and print sizing for the figures of the paper.

reset() is applied at the start of figures.run and gene_level.run, so the result does not depend
on a user's own matplotlibrc or on settings left over from earlier code in the same Python session.

save() writes every figure at Nature Communications print size: 180 mm wide (one full-width
figure), text between 5 and 8 pt, lines scaled with the figure, Arial (or Helvetica). The PDF is
vector with the fonts embedded as TrueType; the PNG is 300 dpi. Where Arial and Helvetica are not
installed, the metric-compatible Liberation Sans is used, so the layout is the same.
"""
import re

import matplotlib
from matplotlib import font_manager
from matplotlib.collections import Collection, QuadMesh
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

FONTS = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
WIDTH_MM = 180          # full page width of Nature Communications
MAX_HEIGHT_MM = 225     # page height available to a figure and its title
FONT_MIN, FONT_MAX = 5.5, 7.5
PNG_DPI = 300

# shorter category names for tick labels, so two panels fit side by side at 180 mm
SHORT_LABELS = {
    "Shortened life span: childhood/adolescence": "Shortened lifespan: child",
    "Shortened life span: adulthood": "Shortened lifespan: adult",
    "Shortened life span: infancy": "Shortened lifespan: infancy",
    "Sensory impairment: touch, others": "Sensory: touch, others",
    "Sensory Impairment – Hearing": "Sensory: hearing",
    "Sensory Impairment – Vision": "Sensory: vision",
    "Sensory Impairment - Hearing": "Sensory: hearing",
    "Sensory Impairment - Vision": "Sensory: vision",
    "Internal physical malformations": "Internal malformations",
    "Out of Scope: Acquired/Multifactorial": "Out of Scope",
    "NFC (Not Further Classified)": "NFC",
    "NFC - Clinical Signs": "Clinical sign/lab",
    "Clinical Sign or Laboratory Abnormality": "Clinical sign/lab",
}

PAPER_RC = {
    "text.hinting": "no_hinting",
    "font.family": "sans-serif",
    "font.sans-serif": FONTS,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 200,
}


def font():
    """The first of FONTS that is installed."""
    for f in FONTS:
        try:
            font_manager.findfont(f, fallback_to_default=False)
            return f
        except ValueError:
            continue
    return "DejaVu Sans"


def reset():
    """Restores matplotlib's built-in defaults, then applies PAPER_RC. Math text (used for the
    exponents of P values) is set in the same font as the rest of the text."""
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(PAPER_RC)
    f = font()
    matplotlib.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": f, "mathtext.sf": f,
                                "mathtext.it": f"{f}:italic", "mathtext.bf": f"{f}:bold"})


PANEL_LETTER = re.compile(r"^([A-H])(?:\.\s*|\s+)(?=\S)")


def _panel_letters(fig):
    """Writes every panel title as 'A  Title': the letter, two spaces, then the title."""
    for ax in fig.get_axes():
        for t in (ax.title, ax._left_title, ax._right_title):
            m = PANEL_LETTER.match(t.get_text())
            if m:
                t.set_text(f"{m.group(1)}  {t.get_text()[m.end():]}")


def _texts(fig):
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def _shorten_ticklabels(ax):
    for axis in (ax.xaxis, ax.yaxis):
        labels = [t.get_text() for t in axis.get_ticklabels()]
        if any(l in SHORT_LABELS for l in labels):
            axis.set_ticks(axis.get_ticklocs(), [SHORT_LABELS.get(l, l) for l in labels])


def _wrap_titles(ax, renderer):
    """Puts the part of a panel title in brackets on a second line when the title is wider than the panel."""
    width = ax.get_window_extent(renderer).width
    for t in (ax.title, ax._left_title, ax._right_title):
        text = t.get_text()
        if " (" in text and "\n" not in text and t.get_window_extent(renderer).width > width:
            t.set_text(text.replace(" (", "\n(", 1))


def _rebuild_legend(ax, mapped):
    """Draws an axes legend again at the print font size (a legend sets the size of its keys when
    it is created). A legend placed outside the axes on the right is moved below the axes, in
    columns, so that the axes keep their width."""
    leg = ax.get_legend()
    if leg is None:
        return
    handles, labels = leg.legend_handles, [t.get_text() for t in leg.texts]
    title = leg.get_title().get_text() or None
    size = mapped(leg._fontsize)
    common = dict(title=title, prop={"family": FONTS, "size": size},
                  title_fontproperties={"family": FONTS, "size": size},
                  handlelength=1.4, handleheight=0.8, columnspacing=1.0, borderpad=0.4)
    anchor = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    if anchor.x0 >= 0.99:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0), bbox_transform=ax.transAxes,
                  borderaxespad=4.2, ncol=min(len(labels), 3), frameon=False, **common)
    else:
        ax.legend(handles, labels, loc=leg._loc, bbox_to_anchor=tuple(anchor.bounds), bbox_transform=ax.transAxes,
                  ncol=leg._ncols, frameon=leg.get_frame_on(), **common)


def to_print_size(fig, width_mm=WIDTH_MM):
    """Rescales a figure drawn at screen size to print size, once.

    The figure keeps its aspect ratio. Font sizes are mapped linearly from the range used in the
    figure to FONT_MIN to FONT_MAX, so titles stay larger than tick labels; line widths, marker
    sizes and scatter areas shrink with the figure but not below what prints legibly."""
    if getattr(fig, "_print_sized", False):
        return
    _panel_letters(fig)
    w, h = fig.get_size_inches()
    target = width_mm / 25.4
    s = target / w
    texts = _texts(fig)
    sizes = [t.get_fontsize() for t in texts]
    lo, hi = (min(sizes), max(sizes)) if sizes else (10, 10)

    def mapped(f):
        f = min(max(f, lo), hi)
        return round((FONT_MIN + FONT_MAX) / 2 if hi == lo else FONT_MIN + (f - lo) / (hi - lo) * (FONT_MAX - FONT_MIN), 2)

    # tick labels are recreated when the figure is drawn, so set them on the axis as well
    tick_sizes = {}
    for ax in fig.get_axes():
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            labels = [t for t in axis.get_ticklabels() if t.get_text()] or axis.get_ticklabels()
            if labels:
                tick_sizes[(ax, name)] = labels[0].get_fontsize()
    for t in texts:
        t.set_fontsize(mapped(t.get_fontsize()))
        t.set_fontfamily(FONTS)
    for (ax, name), f in tick_sizes.items():
        ax.tick_params(axis=name, which="both", labelsize=mapped(f), labelfontfamily=FONTS)
    for ax in fig.get_axes():
        _shorten_ticklabels(ax)
        _rebuild_legend(ax, mapped)
    for leg in fig.legends:
        leg._fontsize = mapped(leg._fontsize)
    for ln in fig.findobj(Line2D):
        ln.set_linewidth(min(1.5, max(0.4, ln.get_linewidth() * s)))
        ln.set_markersize(min(6, max(1.5, ln.get_markersize() * s)))
        ln.set_markeredgewidth(max(0.2, ln.get_markeredgewidth() * s))
    for p in fig.findobj(Patch):
        lw = p.get_linewidth()
        if lw:
            p.set_linewidth(max(0.2, lw * s))
    for c in fig.findobj(Collection):
        lws = c.get_linewidths()
        if len(lws):
            c.set_linewidths([max(0.2, x * s) for x in lws])
        if hasattr(c, "get_sizes") and len(c.get_sizes()):
            c.set_sizes([max(2.0, x * s * s) for x in c.get_sizes()])
    for ax in fig.get_axes():
        for side in ax.spines.values():
            side.set_linewidth(max(0.4, side.get_linewidth() * s))
        ax.tick_params(which="major", width=0.5, length=2.5, pad=2)
        ax.tick_params(which="minor", width=0.4, length=1.5)
        if any(isinstance(c, QuadMesh) for c in ax.collections) and not hasattr(ax, "_colorbar"):
            ax.tick_params(which="both", length=0)   # heatmap: no tick marks beside the cell values
    fig.set_size_inches(target, h * s)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.get_axes():
        _wrap_titles(ax, renderer)
    fig._print_sized = True


def save(fig, path, width_mm=WIDTH_MM, **kwargs):
    """Saves fig at print size: vector PDF, 300 dpi PNG. Extra keyword arguments go to savefig."""
    to_print_size(fig, width_mm)
    path = str(path)
    kwargs.pop("dpi", None)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.02)
    kwargs.setdefault("facecolor", "white")
    if kwargs["bbox_inches"] == "tight":
        fit_width(fig, width_mm)
    fig.savefig(path, dpi=PNG_DPI, **kwargs)
    if path.lower().endswith(".png"):   # RGB, as the journal asks, not RGBA
        from PIL import Image
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        rgb.save(path, dpi=(PNG_DPI, PNG_DPI))


def fit_width(fig, width_mm=WIDTH_MM):
    """Narrows the figure until everything drawn, legends and labels outside the axes included,
    fits in width_mm. Text keeps its size; the axes get narrower."""
    target = width_mm / 25.4 - 2 * 0.02 - 0.01   # savefig adds pad_inches on each side
    for _ in range(8):
        fig.canvas.draw()
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        if bbox.width <= target:
            return
        w, h = fig.get_size_inches()
        fig.set_size_inches(max(w - (bbox.width - target) * 1.05, 1.0), h)
