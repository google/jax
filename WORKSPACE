# The XLA commit is determined by third_party/xla/workspace.bzl.
load("//third_party/xla:workspace.bzl", jax_xla_workspace = "repo")
jax_xla_workspace()

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")
python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")
python_init_repositories(
    requirements = {
        "3.10": "//build:requirements_lock_3_10.txt",
        "3.11": "//build:requirements_lock_3_11.txt",
        "3.12": "//build:requirements_lock_3_12.txt",
        "3.13": "//build:requirements_lock_3_13.txt",
    },
    local_wheel_workspaces = ["//jaxlib:jax.bzl"],
    local_wheel_dist_folder = "../dist",
    default_python_version = "system",
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")
python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")
python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")
install_deps()

# Optional, to facilitate testing against newest versions of Python
load("@xla//third_party/py:python_repo.bzl", "custom_python_interpreter")
custom_python_interpreter(
    name = "python_dev",
    urls = ["https://www.python.org/ftp/python/3.13.0/Python-{version}.tgz"],
    strip_prefix = "Python-{version}",
    version = "3.13.0a6",
)

load("@xla//:workspace4.bzl", "xla_workspace4")
xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")
xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")
xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")
xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")
xla_workspace0()

load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()
