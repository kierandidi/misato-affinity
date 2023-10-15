"""Plot style for publication quality figures"""
from dataclasses import dataclass
from distutils.spawn import find_executable
from typing import Sequence, Tuple

import matplotlib
import matplotlib.style
import numpy as np
import seaborn as sns


@dataclass
class TEXTWIDTHS:
    # Determined via: https://tex.stackexchange.com/a/39384
    #  all textwidths measured in pt
    MRES_REPORT: float = 398.3386
    PHD_THESIS: float = 455.24411
    NEURIPS_ARTICLE: float = 397.48499
    LATEX_ARTICLE: float = 345.0


def matplotlib_defaults(use_tex: bool = True, autoupdate: bool = False) -> None:
    """Apply plotting style to produce nice looking figures.
    Call this at the start of a script which uses `matplotlib`.
    Can enable `matplotlib` LaTeX backend if it is available.
    Args:
        use_tex (bool, optional): Whether or not to use latex matplotlib backend.
            Defaults to True.
    """
    # matplotlib.use('agg') this used to be required for jasmin
    p_general = {
        "font.family": "Computer Modern",  # "STIXGeneral",  # Nice alternative font.
        # "font.family": "serif",
        # "font.serif": [],
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # Set the font for maths
        "mathtext.fontset": "cm",
        # "font.sans-serif": ["DejaVu Sans"],  # gets rid of error messages
        # "font.monospace": [],
        "lines.linewidth": 1.0,
        "scatter.marker": "+",
        "image.cmap": "RdYlBu_r",
        "text.usetex": False,
    }
    if use_tex:
        try:
            # See: https://matplotlib.org/stable/tutorials/text/usetex.html
            print("Finding latex executable...")
            assert find_executable("latex")
            print("Found. Using latex backend.")
            p_tex = {
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "pgf.preamble": (
                    r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc}"
                    + r"\usepackage[separate -uncertainty=true]{siunitx}"
                ),
            }
            p_general.update(p_tex)
        except:
            print("Latex executable not found, deactivating latex backend.")
    if autoupdate:
        matplotlib.rcParams.update(p_general)
    return p_general


def label_subplots(
    axs: Sequence[matplotlib.pyplot.axes],
    labels: Sequence[str] = [chr(ord("`") + z) for z in range(1, 27)],
    start_from: int = 0,
    fontsize: int = 10,
    x_pos: float = 0.02,
    y_pos: float = 0.95,
) -> None:
    """Adds (a), (b), (c) at the top left of each subplot panel.
    Labelling order achieved through ravelling the input list / array.
    Args:
        axs (Sequence[matplotlib.axes]): list or array of subplot axes.
        labels (Sequence[str]): A sequence of labels for the subplots.
        start_from (int, optional): skips first ${start_from} labels. Defaults to 0.
        fontsize (int, optional): Font size for labels. Defaults to 10.
        x_pos (float, optional): Relative x position of labels. Defaults to 0.02.
        y_pos (float, optional): Relative y position of labels. Defaults to 0.95.
    Returns:
        void; alters the `matplotlib.pyplot.axes` objects
    """
    if isinstance(axs, list):
        axs = np.asarray(axs)
    assert len(axs.ravel()) + start_from <= len(labels)
    subset_labels = []
    for i in range(len(axs.ravel())):
        subset_labels.append(labels[i + start_from])
    for i, label in enumerate(subset_labels):
        axs.ravel()[i].text(
            x_pos,
            y_pos,
            str("(" + label + ")"),
            color="black",
            transform=axs.ravel()[i].transAxes,
            fontsize=fontsize,
            fontweight="bold",
            va="top",
        )


def get_dim(
    width: float = TEXTWIDTHS.LATEX_ARTICLE,
    fraction_of_line_width: float = 1,
    ratio: float = (5**0.5 - 1) / 2,
) -> Tuple[float, float]:
    """Return figure height, width in inches to avoid scaling in latex.
       Default is golden ratio, with figur occupying full page width.
    Args:
        width (float): Textwidth of the report to make fontsizes match.
        fraction_of_line_width (float, optional): Fraction of the document width
            which you wish the figure to occupy.  Defaults to 1.
        ratio (float, optional): Fraction of figure width that the figure height
            should be. Defaults to (5 ** 0.5 - 1)/2.
    Returns:
        fig_dim (tuple):
            Dimensions of figure in inches
    """

    # Width of figure
    fig_width_pt = width * fraction_of_line_width

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def set_dim(
    fig: matplotlib.pyplot.figure,
    width: float = TEXTWIDTHS.LATEX_ARTICLE,
    fraction_of_line_width: float = 1,
    ratio: float = (5**0.5 - 1) / 2,
) -> None:
    """Set aesthetic figure dimensions to avoid scaling in latex.
    Args:
        fig (matplotlib.pyplot.figure): Figure object to resize.
        width (float): Textwidth of the report to make fontsizes match.
        fraction_of_line_width (float, optional): Fraction of the document width
            which you wish the figure to occupy.  Defaults to 1.
        ratio (float, optional): Fraction of figure width that the figure height
            should be. Defaults to (5 ** 0.5 - 1)/2.
    Returns:
        void; alters current figure to have the desired dimensions
    """
    fig.set_size_inches(
        get_dim(width=width, fraction_of_line_width=fraction_of_line_width, ratio=ratio)
    )


CAMBRIDGE_COLOURS = {
    "Pantone_197": "#E89CAE",
    "Pantone_284": "#6CACE4",
    "Pantone_142": "#F1BE48",
    "Pantone_583": "#B7BF10",
    "Pantone_5215": "#AF95A6",
    "Pantone_557": "#85B09A",
    "Pantone_199": "#D50032",
    "Pantone_285": "#0072CE",
    "Pantone_158": "#E87722",
    "Pantone_369": "#64A70B",
    "Pantone_513": "#93328E",
    "Pantone_7466": "#00B0B9",
    "Pantone_1955": "#A81538",
    "Pantone_541": "#003C71",
    "Pantone_718": "#BE4D00",
    "Pantone_574": "#4E5B31",
    "Pantone_669": "#3F2A56",
    "Pantone_5473": "#115E67",
    "cambridge_blue": "#a3c1ad",
}
ETH_COLOURS = {
    "BLUE": "#215CAF",
    "PETROL": "#007894",
    "GREEN": "#627313",
    "BRONZE": "#8E6713",
    "RED": "#B7352D",
    "PURPLE": "#A30774",
    "GREY": "#6F6F6F",
}
PALETTES = {
    # Cambridge colour palettes from: https://www.cam.ac.uk/system/files/guidelines_v8_december_2019.pdf#page=17
    # Use itertools.cycle to use the colours in a cycle for matplotlib
    "CAMBRIDGE_LIGHT": sns.color_palette(
        list(
            map(
                lambda x: CAMBRIDGE_COLOURS[x],
                [
                    "Pantone_197",
                    "Pantone_284",
                    "Pantone_142",
                    "Pantone_583",
                    "Pantone_5215",
                    "Pantone_557",
                ],
            )
        )
    ),
    "CAMBRIDGE_CORE": sns.color_palette(
        list(
            map(
                lambda x: CAMBRIDGE_COLOURS[x],
                [
                    "Pantone_199",
                    "Pantone_285",
                    "Pantone_158",
                    "Pantone_369",
                    "Pantone_513",
                    "Pantone_7466",
                ],
            )
        )
    ),
    "CAMBRIDGE_DARK": sns.color_palette(
        list(
            map(
                lambda x: CAMBRIDGE_COLOURS[x],
                [
                    "Pantone_1955",
                    "Pantone_541",
                    "Pantone_718",
                    "Pantone_574",
                    "Pantone_669",
                    "Pantone_5473",
                ],
            )
        )
    ),
    "ETH": sns.color_palette(
        list(
            map(
                lambda x: ETH_COLOURS[x],
                ["BLUE", "PETROL", "GREEN", "BRONZE", "RED", "PURPLE", "GREY"],
            )
        )
    ),
    "DARK_PALETTE": sns.color_palette(
        [
            "#4d2923ff",
            "#494f1fff",
            "#38734bff",
            "#498489ff",
            "#8481baff",
            "#c286b2ff",
            "#d7a4a3ff",
        ]
    ),
}
