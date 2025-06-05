"""
This module defines generic helper functions for qurveros.
"""

import pandas as pd
from qurveros.settings import settings
from qurveros import spacecurve
import sys


def progbar_range(num_iters, title="", depth=0):
    """
    Creates a simple progress bar depending on the depth of the nested
    iterations.

    Args:
        num_iters (int): The number of iterations.
        title (str): The title for the progress bar.
        depth (int): Specifies the position in the tree of nested
        iterations.

    Returns:
        A generator that prints the percentage of the current step with
        respect to the number of iterations.
    """

    if depth > 0:
        title = "\t" * depth + " \\" + "-" * 4 + " " + title

    print("\n", end="", file=sys.stderr)

    for step in range(num_iters):
        percentage = 100 * step / (num_iters)

        print(f"\r{title}:\t{percentage:^6.2f}" + "%", end="", file=sys.stderr)
        yield step

    print(f"\r{title}:\t{100.00:^6.2f}" + "%", end="", file=sys.stderr)
    if depth > 0:
        print("\r" + " " * 2 * len(title), end="", file=sys.stderr)
    else:
        print("\n", end="", file=sys.stderr)


def prepare_bezier_from_file(filename, *, is_barq, n_points=None):
    """
    Creates a BezierCurve instance from a csv file.

    Args:
        filename (str): The name of the file where the data is stored.
        is_barq (bool): If the control points correspond to a BarqCurve,
        the first element is the barq angle for the TTC method.
        n_points (int): The number of points to sample the curve.

    Returns:
        A BezierCurve instance with the frenet_dict initialized and the
        barq_angle appropriately set (if applicable).
    """

    if n_points is None:
        n_points = settings.options["CURVE_POINTS"]

    control_points = pd.read_csv(filename).to_numpy()

    if is_barq:
        barq_angle = control_points[0, 0]
        control_points = control_points[1:, :]

    beziercurve = spacecurve.BezierCurve(control_points)
    beziercurve.evaluate_frenet_dict(n_points)

    if is_barq:
        beziercurve.get_frenet_dict().update(barq_angle=barq_angle)

    return beziercurve
