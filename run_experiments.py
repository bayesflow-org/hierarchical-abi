#!/usr/bin/env python3
import logging
import subprocess

import numpy as np
import pandas as pd

script_name = 'gaussian_flat_score_matching.py'
max_obs = [1, 100]


def setup_logger(log_filename):
    """
    Create and return a logger that logs to both a file and the console.
    Each logger is unique per run based on the log filename.
    """
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers for a clean slate.
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler logs all messages (DEBUG and above).
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Console handler logs only INFO and above.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    return logger


def main():
    # Loop through the different max_obs values and corresponding arguments.
    for m_obs in max_obs:
        # For m_obs==1, 30 experiments; otherwise use 10.
        arguments = np.arange(30 if m_obs == 1 else 10)
        for arg in arguments:
            # Define a unique log file name for this run.
            log_filename = f"logs/script_{m_obs}_{arg}.log"
            logger = setup_logger(log_filename)

            logger.info(f"Running {script_name} with argument: {m_obs}-{arg}")
            try:
                # Run the script and capture stdout/stderr.
                result = subprocess.run(
                    ["python", script_name, str(m_obs), str(arg)],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Log standard output if any.
                if result.stdout:
                    logger.debug("Standard Output:\n" + result.stdout)
                # Log standard error if any.
                if result.stderr:
                    logger.error("Standard Error:\n" + result.stderr)

                # Log based on return code.
                if result.returncode != 0:
                    logger.warning(f"Script failed for argument '{m_obs}-{arg}' with return code {result.returncode}")
                else:
                    logger.info(f"Script succeeded for argument '{m_obs}-{arg}'")

            except Exception as e:
                logger.exception(f"An error occurred while running script with argument '{m_obs}-{arg}': {e}")

            # Clean up handlers to prevent duplicate logs in subsequent iterations.
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


def main2(var_id):
    variable_of_interest = ['mini_batch', 'cosine_shift', 'damping_factor_t',
                            'n_conditions'][var_id]
    if variable_of_interest == 'n_conditions':
        name_i = lambda \
            i: f"model{i}_compositional_score_model_v_variance_preserving_cosine_likelihood_weighting_factorized100"
    else:
        name_i = lambda i: f"model{i}_compositional_score_model_v_variance_preserving_cosine_likelihood_weighting"

    results = []
    for i in range(10):
        name_df = f"plots/{name_i(i)}/df_results_{variable_of_interest}.csv"
        r = pd.read_csv(name_df, index_col=0)
        r['model_id'] = i
        results.append(r)

    results = pd.concat(results, ignore_index=True)
    results.drop("list_steps", axis=1, inplace=True)
    results.to_csv(f"results_{variable_of_interest}.csv")

if __name__ == "__main__":
    #main()
    #print("All runs completed.")
    main2(var_id=0)
    print("Var0 joined.")
    main2(var_id=1)
    print("Var1 joined.")
    main2(var_id=2)
    print("Var2 joined.")
    main2(var_id=3)
    print("Var3 joined.")
