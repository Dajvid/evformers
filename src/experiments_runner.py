import argparse
import errno
import fcntl
import multiprocessing
import subprocess
import time
import json
import jsonschema
import concurrent.futures

from train_model import parse_args as parse_train_args
from evolve import main as evolve_main
from evolve import parse_args as parse_evolve_args
from multiprocessing import Manager

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
    parser.add_argument("--add-new", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--new-file", type=str,
                        help="Path to the file containing the new experiments to add", default="../new_expr.json")
    parser.add_argument("--gpu-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-workers", type=int, default=multiprocessing.cpu_count())
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


def add_experiments(new_expr_fn, experiments_fn):
    with open(experiments_fn, "r+") as experiments_f:
        with open(new_expr_fn, "r+") as f:
            new_experiments = json.loads(f.read())
            jsonschema.validate(new_experiments, schema)
            for experiment in new_experiments:
                if experiment["requires-gpu"]:
                    parse_train_args(experiment["command"][2:])
                else:
                    parse_evolve_args(experiment["command"])
            print("Adding new experiments to the file")
            obtain_file(experiments_f)
            current_experiments = json.loads(experiments_f.read())
            all_experiments = new_experiments + current_experiments
            experiments_f.seek(0)
            experiments_f.truncate(0)
            experiments_f.write(json.dumps(all_experiments, indent=4, sort_keys=True))
            experiments_f.flush()
            release_file(experiments_f)
            f.seek(0)
            f.truncate(0)


def run_only_gpu_experiments(experiments_f):

    while True:
        with open(experiments_f, "r+") as experiments_fh:
            obtain_file(experiments_fh)
            experiments = json.loads(experiments_fh.read())
            experiment_to_run = find_runable_experiment(experiments, gpu_only=True)
            if experiment_to_run is not None:
                experiment_to_run["remaining-runs"] -= 1
                print(f"Running experiment: {experiment_to_run}")
                experiments_fh.seek(0)
                experiments_fh.truncate(0)
                experiments_fh.write(json.dumps(experiments, indent=4, sort_keys=True))
                experiments_fh.flush()
                release_file(experiments_fh)
                experiment_to_run["command"] += ["--run-id", str(experiment_to_run["total-runs"])]
                sub_completed = subprocess.run(experiment_to_run["command"], stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT)
                if sub_completed.returncode != 0:
                    print(f"Error running experiment: {experiment_to_run}\n{sub_completed.stdout}")
                else:
                    print(f"Experiment ran successfully: {experiment_to_run}")
            else:
                release_file(experiments_fh)
                time.sleep(30)


def get_task(experiments_fh, gpu_only=False):
    obtain_file(experiments_fh)
    f_content = experiments_fh.read()
    if f_content:
        print("Empty file          ")
        experiments = json.loads(f_content)
        experiment_to_run = find_runable_experiment(experiments, gpu_only=gpu_only)
    else:
        experiment_to_run = None
    if experiment_to_run is not None:
        experiment_to_run["remaining-runs"] -= 1
        print(f"Running experiment: {experiment_to_run}")
        experiments_fh.seek(0)
        experiments_fh.truncate(0)
        experiments_fh.write(json.dumps(experiments, indent=4, sort_keys=True))
        experiments_fh.flush()
        experiments_fh.seek(0)
        release_file(experiments_fh)
        return experiment_to_run
    else:
        experiments_fh.flush()
        experiments_fh.seek(0)
        release_file(experiments_fh)
        return None


class CustomProcessPoolExecutor(concurrent.futures.ProcessPoolExecutor):
    def __init__(self, max_workers):
        super().__init__()
        self.manager = Manager()
        self.idle_workers = self.manager.Value('i', max_workers)
        self.lock = self.manager.Lock()

    def submit(self, *args, **kwargs):
        with self.lock:
            self.idle_workers.value -= 1  # Decrease idle worker count when submitting a task
        future = super().submit(*args, **kwargs)
        future.add_done_callback(self._task_done_callback)
        return future

    def _task_done_callback(self, future):
        with self.lock:
            self.idle_workers.value += 1  # Increase idle worker count when a task is done

    def get_idle_workers_count(self):
        with self.lock:
            return self.idle_workers.value


def run_cpu_paralell_experiments(experiments_f, max_workers):
    with CustomProcessPoolExecutor(max_workers) as executor:
        futures = set()
        while True:
            with open(experiments_f, "r+") as experiments_fh:
                while executor.idle_workers.value > 0:
                    print(f"idle workers: {executor.idle_workers.value}\r", end="")
                    experiment = get_task(experiments_fh, gpu_only=False)
                    if experiment is not None:
                        experiment["command"] += ["--run-id", str(experiment["remaining-runs"])]
                        parse_evolve_args(experiment["command"])
                        futures.add(executor.submit(evolve_main, experiment["command"]))
                    else:
                        print(f"Waiting for more experiments, idle workers: {executor.idle_workers.value}\r", end="")
                        time.sleep(30)

                _, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)


def main(argv=None):
    args = parse_args(argv)
    new_expr_fn = args.new_file
    print(f"Running experiments from file: {args.experiments_file}")

    if args.add_new:
        print(f"Adding new experiments from file: {new_expr_fn} to {args.experiments_file}")
        add_experiments(new_expr_fn, args.experiments_file)

    elif args.gpu_only:
        print("Running only experiments that require GPU")
        run_only_gpu_experiments(args.experiments_file)

    elif not args.gpu_only:
        print("Running CPU experiments in parallel")
        run_cpu_paralell_experiments(args.experiments_file, args.max_workers)


if __name__ == '__main__':
    main()
