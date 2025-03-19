#!/usr/bin/env python3
import subprocess
import sys
import numpy as np

script_name = 'gaussian flat score matching.py'
max_obs = [1, 100]

def main():
    for m_obs in max_obs:
        print(m_obs)
        arguments = np.arange(30 if m_obs==1 else 40)
        for arg in arguments:
            print(f"Running {script_name} with argument: {arg}")
            try:
                # Run the other script with the current argument.
                # 'check=False' ensures that subprocess.run does not raise an exception
                # if the child process returns a non-zero exit status.
                result = subprocess.run(["python", script_name, arg], check=False)

                # Check if the script returned a non-zero exit code.
                if result.returncode != 0:
                    print(f"Warning: script failed for argument '{arg}' with return code {result.returncode}",
                          file=sys.stderr)
                else:
                    print(f"script succeeded for argument '{arg}'")
            except Exception as e:
                # Catch any other exceptions and continue.
                print(f"An error occurred while running script with argument '{arg}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
