import argparse
import errno
import fcntl
import subprocess
import time
import json
import jsonschema

# Experiment to run entry example:

schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "command": {"type": "array", "items": {"type": "string"}},
            "total-runs": {"type": "integer"},
            "remaining-runs": {"type": "integer"},
            "requires-gpu": {"type": "boolean"},
        },
        "required": ["command", "total-runs", "remaining-runs", "requires-gpu"]
    }
}


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Tool to automatically run experiments on multiple devices and threads"
    )

    parser.add_argument("--experiments-file", type=str,
                        help="Path to the file containing the experiments to run", default="../experiments.json")
    parser.add_argument("--add-new", type=str, default="None")
    parser.add_argument("--gpu-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def obtain_file(f):
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except IOError as e:
            # raise on unrelated IOErrors
            if e.errno != errno.EAGAIN:
                raise
            else:
                time.sleep(0.5)
    return f


def release_file(f):
    fcntl.flock(f, fcntl.LOCK_UN)


def find_runable_experiment(experiments, gpu_only):
    for experiment in experiments:
        if experiment["remaining-runs"] > 0 and (not gpu_only or experiment["requires-gpu"]):
            return experiment
    return None


def add_experiments(args):
    with open(args.experiments_file, "r+") as experiments_f:
        with open(args.add_new, "r+") as f:
            new_experiments = json.loads(f.read())
            jsonschema.validate(new_experiments, schema)
            print("Adding new experiments to the file")
            obtain_file(experiments_f)
            current_experiments = json.loads(experiments_f.read())
            all_experiments = new_experiments + current_experiments
            experiments_f.seek(0)
            experiments_f.truncate(0)
            experiments_f.write(json.dumps(all_experiments, indent=4, sort_keys=True))
            release_file(experiments_f)
            f.seek(0)
            f.truncate(0)


def run_only_gpu_experiments(experiments_f):
    while True:
        obtain_file(experiments_f)
        experiments = json.loads(experiments_f.read())
        experiment_to_run = find_runable_experiment(experiments, gpu_only=True)
        if experiment_to_run is not None:
            experiment_to_run["remaining-runs"] -= 1
            print(f"Running experiment: {experiment_to_run}")
            experiments_f.seek(0)
            experiments_f.truncate(0)
            experiments_f.write(json.dumps(experiments, indent=4, sort_keys=True))
            release_file(experiments_f)
            experiment_to_run["command"] += ["--run-id", str(experiment_to_run["total-runs"])]
            sub_completed = subprocess.run(experiment_to_run["command"], capture_output=True, stderr=subprocess.STDOUT)
            if sub_completed.returncode != 0:
                print(f"Error running experiment: {experiment_to_run}\n{sub_completed.stdout}")
            else:
                print(f"Experiment ran successfully: {experiment_to_run}")
        else:
            time.sleep(60)


def main(argv=None):
    args = parse_args(argv)
    print(f"Running experiments from file: {args.experiments_file}")

    if args.add_new != "None":
        print(f"Adding new experiments from file: {args.add_new}")
        add_experiments(args)

    elif args.gpu_only:
        print("Running only experiments that require GPU")
        run_only_gpu_experiments(args.experiments_file)

    elif not args.gpu_only:
        print("Running CPU experiments")


if __name__ == '__main__':
    main()
