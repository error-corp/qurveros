import matplotlib

# Selections inspired and adapted from
# https://github.com/hosilva/physrev_mplstyle/blob/main/physrev.mplstyle


# Legend
matplotlib.rcParams["legend.frameon"] = False
matplotlib.rcParams["legend.fontsize"] = 9
matplotlib.rcParams["legend.handlelength"] = 1.375
matplotlib.rcParams["legend.labelspacing"] = 0.4
matplotlib.rcParams["legend.columnspacing"] = 1
matplotlib.rcParams["legend.facecolor"] = "white"
matplotlib.rcParams["legend.edgecolor"] = "white"
matplotlib.rcParams["legend.framealpha"] = 1
matplotlib.rcParams["legend.title_fontsize"] = 9

# Figure
matplotlib.rcParams["figure.figsize"] = 3.25, 0.75 * 3.25
matplotlib.rcParams["figure.subplot.left"] = 0.125
matplotlib.rcParams["figure.subplot.bottom"] = 0.175
matplotlib.rcParams["figure.subplot.top"] = 0.95
matplotlib.rcParams["figure.subplot.right"] = 0.95
matplotlib.rcParams["figure.autolayout"] = False

matplotlib.rcParams["lines.linewidth"] = 1
matplotlib.rcParams["lines.markersize"] = 10
matplotlib.rcParams["hatch.linewidth"] = 0.25
matplotlib.rcParams["patch.antialiased"] = True

matplotlib.rcParams["axes.titlesize"] = "large"
matplotlib.rcParams["axes.labelsize"] = 9
matplotlib.rcParams["axes.formatter.use_mathtext"] = True
matplotlib.rcParams["axes.linewidth"] = 0.5

matplotlib.rcParams["grid.linewidth"] = 0.5
matplotlib.rcParams["grid.linestyle"] = "dashed"
matplotlib.rcParams["grid.color"] = "xkcd:light gray"

# Fonts
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "cmr10"

# For saving
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.pad_inches"] = 0.05
matplotlib.rcParams["savefig.dpi"] = 750
matplotlib.rcParams["savefig.transparent"] = True


# Ticks X
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["xtick.bottom"] = True
matplotlib.rcParams["xtick.major.size"] = 3.0
matplotlib.rcParams["xtick.minor.size"] = 1.5
matplotlib.rcParams["xtick.major.width"] = 0.5
matplotlib.rcParams["xtick.minor.width"] = 0.5
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["xtick.minor.visible"] = True
matplotlib.rcParams["xtick.major.top"] = True
matplotlib.rcParams["xtick.major.bottom"] = True
matplotlib.rcParams["xtick.minor.top"] = True
matplotlib.rcParams["xtick.minor.bottom"] = True
matplotlib.rcParams["xtick.major.pad"] = 5.0
matplotlib.rcParams["xtick.minor.pad"] = 5.0
matplotlib.rcParams["ytick.labelsize"] = 9

# Ticks Y
matplotlib.rcParams["ytick.left"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["ytick.major.size"] = 3.0
matplotlib.rcParams["ytick.minor.size"] = 1.5
matplotlib.rcParams["ytick.major.width"] = 0.5
matplotlib.rcParams["ytick.minor.width"] = 0.5
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["ytick.minor.visible"] = True
matplotlib.rcParams["ytick.major.left"] = True
matplotlib.rcParams["ytick.major.right"] = True
matplotlib.rcParams["ytick.minor.left"] = True
matplotlib.rcParams["ytick.minor.right"] = True
matplotlib.rcParams["ytick.major.pad"] = 5.0
matplotlib.rcParams["ytick.minor.pad"] = 5.0
matplotlib.rcParams["xtick.labelsize"] = 9
