import os
import sys
import subprocess


def main():
    """
    Convenience wrapper over :py:mod:`ecoli.experiments.ecoli_master_sim`.
    """
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "ecoli",
        "experiments",
        "ecoli_master_sim.py",
    )
    # Forward all arguments
    cmd = [sys.executable, script_path] + sys.argv[1:]
    # Execute and forward exit code
    proc = subprocess.Popen(cmd)
    proc.wait()
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
