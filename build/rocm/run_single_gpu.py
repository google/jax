#!/usr/bin/env python3
# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

GPU_LOCK = threading.Lock()
LAST_CODE = 0
base_dir = "./logs"

def extract_filename(path):
  base_name = os.path.basename(path)
  file_name, _ = os.path.splitext(base_name)
  return file_name

def generate_final_report(shell=False, env_vars={}):
  env = os.environ
  env = {**env, **env_vars}
  cmd = ["pytest_html_merger", "-i", f'{base_dir}', "-o", f'{base_dir}/final_compiled_report.html']
  result = subprocess.run(cmd,
                          shell=shell,
                          capture_output=True,
                          env=env)
  if result.returncode != 0:
    print("FAILED - {}".format(" ".join(cmd)))
    print(result.stderr.decode())
    # sys.exit(result.returncode)
  return result.returncode, result.stderr.decode(), result.stdout.decode()


def run_shell_command(cmd, shell=False, env_vars={}):
  env = os.environ
  env = {**env, **env_vars}
  result = subprocess.run(cmd,
                          shell=shell,
                          capture_output=True,
                          env=env)
  if result.returncode != 0:
    print("FAILED - {}".format(" ".join(cmd)))
    print(result.stderr.decode())
    # sys.exit(result.returncode)
  return result.returncode, result.stderr.decode(), result.stdout.decode()


def collect_testmodules():
  all_test_files = []
  return_code, stderr, stdout = run_shell_command(
      ["python3", "-m", "pytest", "--collect-only", "tests"])
  if return_code != 0:
    print(stdout)
    print(stderr)
    print("Test module discovery failed.")
    exit(return_code)
  for line in stdout.split("\n"):
    match = re.match("<Module (.*)>", line)
    if match:
      test_file = match.group(1)
      all_test_files.append(test_file)
  print("---------- collected test modules ----------")
  print("Found %d test modules." % (len(all_test_files)))
  print("\n".join(all_test_files))
  print("--------------------------------------------")
  return all_test_files


def run_test(testmodule, gpu_tokens):
  global LAST_CODE
  with GPU_LOCK:
    if LAST_CODE != 0:
      return
    target_gpu = gpu_tokens.pop()
  env_vars = {
      "HIP_VISIBLE_DEVICES": str(target_gpu),
      "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
  }
  testfile = extract_filename(testmodule)
  cmd = ["python3", "-m", "pytest", f'--html={base_dir}/{testfile}_log.html', "--reruns", "3", "-x", testmodule]
  return_code, stderr, stdout = run_shell_command(cmd, env_vars=env_vars)
  with GPU_LOCK:
    gpu_tokens.append(target_gpu)
    if LAST_CODE == 0:
      print("Running tests in module %s on GPU %d:" % (testmodule, target_gpu))
      print(stdout)
      print(stderr)
      LAST_CODE = return_code
  return


def run_parallel(all_testmodules, p):
  print("Running tests with parallelism=", p)
  available_gpu_tokens = list(range(p))
  executor = ThreadPoolExecutor(max_workers=p)
  # walking through test modules
  for testmodule in all_testmodules:
    executor.submit(run_test, testmodule, available_gpu_tokens)
  # waiting for all modules to finish
  executor.shutdown(wait=True)  # wait for all jobs to finish
  return


def find_num_gpus():
  cmd = ["lspci|grep 'controller'|grep 'AMD/ATI'|wc -l"]
  _, _, stdout = run_shell_command(cmd, shell=True)
  return int(stdout)


def main(args):
  all_testmodules = collect_testmodules()
  run_parallel(all_testmodules, args.parallel)
  generate_final_report()
  exit(LAST_CODE)


if __name__ == '__main__':
  os.environ['HSA_TOOLS_LIB'] = "libroctracer64.so"
  parser = argparse.ArgumentParser()
  parser.add_argument("-p",
                      "--parallel",
                      type=int,
                      help="number of tests to run in parallel")
  args = parser.parse_args()
  if args.parallel is None:
    sys_gpu_count = find_num_gpus()
    args.parallel = sys_gpu_count
    print("%d GPUs detected." % sys_gpu_count)

  main(args)
