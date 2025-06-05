"""
This module contains the settings for qurveros.
Additional settings or modifications of the existing ones can be achieved
by modifying the file qurveros_settings.json.
"""

import json
import os


class Settings:

    filename = 'qurveros_settings.json'

    options = {}

    # Number of points to sample along the parameter interval
    # Depending on the bandwidth of the resulting pulses, the curve might
    # require more points.
    options['CURVE_POINTS'] = 4096

    # Simulation points
    options['SIM_POINTS'] = 4096

    # Qutip options
    options['QUTIP_OPTIONS'] = {'max_step': 0.5/options['SIM_POINTS'],
                                'atol': 1e-14,
                                'rtol': 1e-14}

    # Number of noise realizations
    options['NUM_REALIZATIONS'] = 200

    # Fraction of filter terms with respect to the number of samples.
    # See qubit_bench/noisetools.py
    options['FIR_TERMS_FRACTION'] = 0.9

    # Noise experiments
    options['DELTA_Z_INTERVAL'] = (-3, 0)
    options['EPSILON_INTERVAL'] = (-4, -1)
    options['MAX_POINTS'] = 12
    options['COUNTOUR_LEVELS'] = (-9, -8, -7, -6, -5)

    # The number of derivatives used for the moving frame.
    # The value is chosen so that inflection points are appropriately handled.
    options['NUM_DERIVS'] = 5

    # The criterion to detect an inflection point.
    options['INFLECTION_NORM'] = 1e-3

    # The minimum norm for the pgf_parameters (BARQ).
    # See barqtools.py for details.
    options['FIX_NORM'] = 1e-2

    # Expected number of singular points.
    # See calculate_singularity_indices() in frametools.py
    options['NUM_SINGLS'] = 5

    # Total torsion compensation 2k\pi range search.
    # See calculate_ttc_detuning() in frametools.py
    options['ANGLE_K_MAX'] = 3

    # Control dictionary fields' entries names.
    options['FIELD_NAMES'] = ['omega', 'phi', 'delta']

    # Number of points to sample along the parameter interval for
    # optimization.
    options['OPT_POINTS'] = 4096

    # Plotting
    options['CURVE_COLORS'] = ["#929A9C", "#E7ACAE", "#921417"]
    options['FIELD_COLORS'] = ["#000000", "#305CDE", "#A67B5B"]
    options['COUNTOUR_COLORS'] = ["#09B8B8", "#5B5585"]

    # Number of tries to search for the qurveros_settings.json
    # up the file tree.
    NUM_TRIES = 3

    def __init__(self):

        self.read_from_file()
        self.options.update(self.options_from_file)
        self._DEPHASING_PROGBAR_DEPTH = 0

    def get_options(self):
        return self.options

    def __repr__(self):

        for option_name, value in self.options.items():

            print(f"{option_name:^20}: \t {value}")

        return ''

    def read_from_file(self):

        filename = self.filename

        path = os.getcwd()

        for _ in range(self.NUM_TRIES):

            filepath = os.path.join(path, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    options_from_file = json.load(file)
            except FileNotFoundError:
                options_from_file = {}
                path = os.path.dirname(path)

            self.options_from_file = options_from_file


settings = Settings()
