import qutip
import numpy as np
import logging
import os
import pandas as pd
import argparse
import datetime

from qurveros import misctools
from qurveros.qubit_bench import noise_experiments


def run_td_sim(filename, alpha, num_realizations, seed=None):

    logger = logging.getLogger('TDsim')

    logging.basicConfig(
        filename='sims/tdsim.log',
        encoding='utf-8',
        level=logging.INFO,
        format='%(asctime)s %(message)s')

    logger.info(f'Simulation for {filename} and alpha={alpha}')

    if 'hadamard' in filename:
        u_target = 1/np.sqrt(2)*qutip.Qobj([[1, 1], [1, -1]])
    elif 'xgate' in filename:
        u_target = qutip.sigmax()
    else:
        u_target = None

    path = os.path.join(os.getcwd(), 'control_points', filename)
    curve = misctools.prepare_bezier_from_file(path, is_barq=True)
    curve.evaluate_control_dict('TTC')

    if seed is None:
        seed = 0

    rng = np.random.default_rng(seed)

    sim_dict = noise_experiments.td_dephasing_experiment(
                    curve.get_control_dict(),
                    u_target,
                    alpha,
                    rng,
                    num_realizations=num_realizations)

    data_to_save = np.vstack([
        sim_dict['tg_delta_z'],
        sim_dict['infidelity_matrix']
    ])

    df = pd.DataFrame(data_to_save)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    filename_for_save = \
        f'sims/{timestamp}_{filename.split('.')[0]}_noise_{alpha}.csv'
    df.to_csv(filename_for_save,
              index=False,
              float_format='%.12f',
              na_rep='nan')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Time dependent simulation runner.')
    parser.add_argument('filename', help='Control points filename.')

    parser.add_argument('alpha', help='The noise PSD exponent.', type=int)
    parser.add_argument('-n', '--num_rel',
                        help='Number of noise realizations.',
                        type=int)

    args = parser.parse_args()

    run_td_sim(args.filename, args.alpha, args.num_rel)
