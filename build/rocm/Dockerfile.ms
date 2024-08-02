################################################################################
FROM ubuntu:20.04 AS rocm_base
################################################################################

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y python3

# Add target file to help determine which device(s) to build for
ARG GPU_DEVICE_TARGETS="gfx900 gfx906 gfx908 gfx90a gfx940 gfx941 gfx942 gfx1030 gfx1100"
ENV GPU_DEVICE_TARGETS=${GPU_DEVICE_TARGETS}

# Install ROCM
ARG ROCM_VERSION=6.0.0
ARG ROCM_PATH=/opt/rocm-${ROCM_VERSION}
ENV ROCM_PATH=${ROCM_PATH}
ARG ROCM_BUILD_JOB
ARG ROCM_BUILD_NUM
RUN --mount=type=bind,source=build/rocm/tools/get_rocm.py,target=get_rocm.py \
    python3 get_rocm.py --rocm-version=$ROCM_VERSION --job-name=$ROCM_BUILD_JOB --build-num=$ROCM_BUILD_NUM

# Set up paths
ENV HCC_HOME=$ROCM_PATH/hcc
ENV HIP_PATH=$ROCM_PATH/
ENV OPENCL_ROOT=$ROCM_PATH/opencl
ENV PATH="$HCC_HOME/bin:$HIP_PATH/bin:${PATH}"
ENV PATH="$ROCM_PATH/bin:${PATH}"
ENV PATH="$OPENCL_ROOT/bin:${PATH}"
ENV PATH="/root/bin:/root/.local/bin:$PATH"

# install pyenv dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y git libssl-dev

# Install pyenv with different python versions
ARG PYTHON_VERSION=3.10.14
RUN git clone https://github.com/pyenv/pyenv.git /pyenv
ENV PYENV_ROOT /pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv install $PYTHON_VERSION
RUN eval "$(pyenv init -)" && \
    pyenv local ${PYTHON_VERSION} && \
    pip3 install --upgrade --force-reinstall setuptools pip && \
    pip install \
        numpy setuptools build wheel six auditwheel scipy \
        pytest pytest-html pytest_html_merger pytest-reportlog \
        pytest-rerunfailures matplotlib absl-py \
        flatbuffers hypothesis


################################################################################
FROM rocm_base AS rt_build
################################################################################

ARG JAX_VERSION
ARG JAX_COMMIT
ARG XLA_COMMIT

LABEL com.amdgpu.rocm_version="$ROCM_VERSION" \
      com.amdgpu.python_version="$PYTHON_VERSION" \
      com.amdgpu.jax_version="$JAX_VERSION" \
      com.amdgpu.jax_commit="$JAX_COMMIT" \
      com.amdgpu.xla_commit="$XLA_COMMIT"

RUN --mount=type=bind,source=wheelhouse,target=/wheelhouse \
    pip install --find-links /wheelhouse jax jaxlib jax_rocm60_plugin jax_rocm60_pjrt
