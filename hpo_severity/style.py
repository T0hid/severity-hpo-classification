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
import matplotlib.axes
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

# category names on the figures, as written in the manuscript
SHORT_LABELS = {
    "Sensory impairment: touch, others": "Sensory impairment: touch, others",
    "Sensory Impairment – Hearing": "Sensory impairment: hearing",
    "Sensory Impairment – Vision": "Sensory impairment: vision",
    "Sensory Impairment - Hearing": "Sensory impairment: hearing",
    "Sensory Impairment - Vision": "Sensory impairment: vision",
    "Out of Scope: Acquired/Multifactorial": "Out of Scope",
    "NFC (Not Further Classified)": "NFC",
    "NFC - Clinical Signs": "Clinical Sign or Laboratory Abnormality",
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


PANEL_LETTER = re.compile(r"^([A-Ha-h])(?:\.\s*|\s+)(?=\S)")


def _panel_letters(fig):
    """Writes every panel title as 'a  Title': the lower-case letter, two spaces, then the title."""
    for ax in fig.get_axes():
        for t in (ax.title, ax._left_title, ax._right_title):
            m = PANEL_LETTER.match(t.get_text())
            if m:
                t.set_text(f"{m.group(1).lower()}  {t.get_text()[m.end():]}")


def _texts(fig):
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def wrap_label(label, width=24):
    """Puts a long category name on two lines: after the colon, or at the space nearest the middle."""
    if "\n" in label or len(label) <= width:
        return label
    if ": " in label:
        return label.replace(": ", ":\n", 1)
    spaces = [i for i, c in enumerate(label) if c == " "]
    if not spaces:
        return label
    i = min(spaces, key=lambda k: abs(k - len(label) / 2))
    return label[:i] + "\n" + label[i + 1:]


def set_category_labels(ax, axis="y", wrap=True):
    """Writes the category tick labels of an axis with the manuscript names, long names on two lines."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    labels = [SHORT_LABELS.get(t.get_text(), t.get_text()) for t in a.get_ticklabels()]
    a.set_ticks(a.get_ticklocs(), [wrap_label(l) if wrap else l for l in labels])


def _shorten_ticklabels(ax):
    for axis in (ax.xaxis, ax.yaxis):
        labels = [t.get_text() for t in axis.get_ticklabels()]
        if any(l in SHORT_LABELS for l in labels):
            axis.set_ticks(axis.get_ticklocs(), [SHORT_LABELS.get(l, l) for l in labels])


def _all_axes(fig):
    return [o for o in fig.findobj(lambda o: isinstance(o, matplotlib.axes.Axes))]


def _title_room(ax, t, renderer):
    """Width available to a panel title: from its left end to the right edge of the figure, or to the
    nearest title or label of another panel at the same height."""
    fig = ax.get_figure(root=True)
    box = t.get_window_extent(renderer)
    right = fig.bbox.x1 - 4
    for other in _all_axes(fig):
        if other is ax:
            continue
        texts = [other.title, other._left_title, other._right_title, other.yaxis.label]
        texts += list(other.get_yticklabels())
        for x in texts:
            if not x.get_visible() or not x.get_text().strip():
                continue
            o = x.get_window_extent(renderer)
            if o.x0 > box.x0 + 1 and o.y0 < box.y1 and o.y1 > box.y0:
                right = min(right, o.x0 - 8)
    return right - box.x0


def _wrap_titles(ax, renderer):
    """Breaks a panel title that would run into another panel or past the figure edge: first before
    the part in brackets, then between words."""
    for t in (ax.title, ax._left_title, ax._right_title):
        text = t.get_text()
        if not text.strip() or t.get_window_extent(renderer).width <= _title_room(ax, t, renderer):
            continue
        if " (" in text and "\n" not in text:
            t.set_text(text.replace(" (", "\n(", 1))
            if t.get_window_extent(renderer).width <= _title_room(ax, t, renderer):
                continue
        room = _title_room(ax, t, renderer)
        lines = []
        for part in t.get_text().split("\n"):
            line = ""
            for word in part.split(" "):
                trial = f"{line} {word}" if line else word
                t.set_text(trial)
                if line and t.get_window_extent(renderer).width > room:
                    lines.append(line)
                    line = word
                else:
                    line = trial
            lines.append(line)
        t.set_text("\n".join(lines))


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
        for ax in fig.get_axes():
            _wrap_titles(ax, fig.canvas.get_renderer())
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        if bbox.width <= target:
            return
        w, h = fig.get_size_inches()
        fig.set_size_inches(max(w - (bbox.width - target) * 1.05, 1.0), h)
