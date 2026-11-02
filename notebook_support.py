"""
notebook_support.py - tiny helper used ONLY by the notebooks (the pipeline scripts do not import it).

It solves three practical problems and contains no modelling logic:
  1. find the project root no matter where the Jupyter kernel was started;
  2. find the four CSVs even if they sit one folder above the project, or use hyphens
     (train-test.csv) instead of the underscores the scripts expect (train_test.csv);
  3. keep one colour palette / plotting style and one place for figure + output folders.

All modelling code lives in common.py and train_and_predict.py and is imported by the notebooks.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TEAL, GREY, RED, ORANGE = "#064A56", "#7F8C8D", "#C0392B", "#E67E22"
PALETTE = [TEAL, ORANGE, RED, GREY]

# canonical name used by common.load_data  ->  names the file may have on disk
DATA_ALIASES = {
    "train_test.csv": ["train_test.csv", "train-test.csv"],
    "validation.csv": ["validation.csv"],
    "validation_predictions_template.csv": ["validation_predictions_template.csv", "validation-predictions-template.csv"],
    "december_chart_inputs.csv": ["december_chart_inputs.csv", "december-chart-inputs.csv"],
}


def project_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: cwd) until a folder containing common.py is found."""
    here = (start or Path.cwd()).resolve()
    for base in [here, *here.parents]:
        if (base / "common.py").is_file():
            return base
    for child in sorted(here.glob("*/common.py")):          # kernel started in the folder ABOVE the project
        return child.parent
    raise FileNotFoundError(f"common.py not found in or above {here}")


def locate_data(root: Path) -> Path:
    """Return a folder containing the four CSVs under the names common.load_data expects.

    Search order: root/data, root, root's parent, parent/data.  Files that exist under a
    different (hyphenated) name are COPIED into root/data - originals are never moved or modified.
    """
    target = root / "data"
    search = [root / "data", root, root.parent, root.parent / "data"]
    for canonical, aliases in DATA_ALIASES.items():
        if (target / canonical).is_file():
            continue
        source = next((d / a for d in search for a in aliases if (d / a).is_file()), None)
        if source is None:
            raise FileNotFoundError(f"Could not find {canonical} (looked for {aliases} in {[str(s) for s in search]})")
        target.mkdir(exist_ok=True)
        shutil.copyfile(source, target / canonical)
        print(f"[notebook_support] copied {source} -> {target / canonical}")
    return target


def setup(notebook_name: str) -> dict:
    """Create output folders and apply the plotting style.  Returns a dict of paths."""
    root = project_root()
    paths = dict(
        root=root,
        data=locate_data(root),
        figures=root / "figures" / "notebooks",
        outputs=root / "outputs" / "notebooks",
    )
    for key in ("figures", "outputs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True,
        "grid.color": "#E3E8EA", "grid.linewidth": 0.7, "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
    pd.set_option("display.width", 160, "display.max_columns", 40, "display.float_format", "{:,.4f}".format)
    paths["name"] = notebook_name
    return paths


def save_fig(fig, paths: dict, name: str) -> Path:
    out = paths["figures"] / f"{paths['name']}_{name}.png"
    fig.savefig(out, bbox_inches="tight")
    return out
