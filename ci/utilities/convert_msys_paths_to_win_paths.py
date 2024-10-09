# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Converts MSYS Linux-like paths stored in env variables to Windows paths.

This is necessary on Windows, because some applications do not understand/handle
Linux-like paths MSYS uses, for example, Docker.
"""
import argparse
import os
import subprocess

def msys_to_windows_path(msys_path):
  """Converts an MSYS path to a Windows path using cygpath.

  Args:
    msys_path: The MSYS path to convert.

  Returns:
    The corresponding Windows path.
  """
  try:
    # Use cygpath with the -w flag to convert to Windows format
    process = subprocess.run(['cygpath', '-w', msys_path], capture_output=True, text=True, check=True)
    windows_path = process.stdout.strip()
    return windows_path
  except FileNotFoundError:
    print("Error: cygpath not found. Make sure it's in your PATH.")
    return None
  except subprocess.CalledProcessError as e:
    print(f"Error converting path: {e}")
    return None

def main(parsed_args: argparse.Namespace):
  converted_paths = {}

  for var, value in os.environ.items():
    if parsed_args.blacklist and var in parsed_args.blacklist:
      continue
    if "_DIR" in var or (args.whitelist and var in parsed_args.whitelist):
      converted_path = msys_to_windows_path(value)
      converted_paths[var] = converted_path

  var_str = '\n'.join(f'export {k}="{v}"'
                      for k, v in converted_paths.items())
  # The string can then be piped into `source`, to re-set the
  # 'converted' variables.
  print(var_str)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=(
      'Convert MSYS paths in environment variables to Windows paths.'))
  parser.add_argument('--blacklist',
                      nargs='*',
                      help='List of variables to ignore')
  parser.add_argument('--whitelist',
                      nargs='*',
                      help='List of variables to include')
  args = parser.parse_args()

  main(args)
