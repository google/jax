import os
import sys

RUNFILES_SUFFIX = "runfiles"

if __name__ == '__main__':
    working_directory = os.getcwd()
    runfiles_directory = working_directory[:working_directory.find(RUNFILES_SUFFIX)+len(RUNFILES_SUFFIX)]
    relative_cuda_lib_paths = os.environ['RELATIVE_CUDA_LIB_PATHS']
    # Format of RELATIVE_CUDA_LIB_PATHS: "repo1_name:/path/to/lib1.so path/to/lib2.so;repo2_name:/path_to_lib3.so"
    ld_library_path = ""
    for repo_libs in relative_cuda_lib_paths.split(";"):
        if repo_libs:
            (repo_name, libs) = repo_libs.split(":")
            first_relative_lib_path = libs.split(" ")[0]
            lib_path = first_relative_lib_path[first_relative_lib_path.find(repo_name):]
            ld_library_path += os.path.dirname(os.path.join(runfiles_directory, lib_path)) + ":"
    os.environ['LD_LIBRARY_PATH'] = ld_library_path
    entry_point_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "%{test_file}")
    args = [sys.executable, entry_point_script] + sys.argv[1:]
    os.execv(args[0], args)