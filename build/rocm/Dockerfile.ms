################################################################################
ARG BASE_DOCKER=ubuntu:20.04
FROM $BASE_DOCKER as rt_build
################################################################################

# Add target file to help determine which device(s) to build for
ARG GPU_DEVICE_TARGETS="gfx900 gfx906 gfx908 gfx90a gfx940 gfx941 gfx942 gfx1030 gfx1100"
ENV GPU_DEVICE_TARGETS=${GPU_DEVICE_TARGETS}

# Install ROCM
ARG ROCM_VERSION=6.0.0
ARG CUSTOM_INSTALL
ARG ROCM_PATH=/opt/rocm-${ROCM_VERSION}
ENV ROCM_PATH=${ROCM_PATH}
COPY ${CUSTOM_INSTALL} /${CUSTOM_INSTALL}
COPY setup.rocm.sh /setup.rocm.sh
RUN /setup.rocm.sh $ROCM_VERSION

# Set up paths
ENV HCC_HOME=$ROCM_PATH/hcc
ENV HIP_PATH=$ROCM_PATH/
ENV OPENCL_ROOT=$ROCM_PATH/opencl
ENV PATH="$HCC_HOME/bin:$HIP_PATH/bin:${PATH}"
ENV PATH="$ROCM_PATH/bin:${PATH}"
ENV PATH="$OPENCL_ROOT/bin:${PATH}"
ENV PATH="/root/bin:/root/.local/bin:$PATH"


# Install pyenv with different python versions
ARG PYTHON_VERSION=3.10.0
RUN git clone https://github.com/pyenv/pyenv.git /pyenv
ENV PYENV_ROOT /pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv install $PYTHON_VERSION
RUN eval "$(pyenv init -)" && pyenv local ${PYTHON_VERSION} && pip3 install --upgrade --force-reinstall setuptools pip && pip install numpy setuptools build wheel six auditwheel scipy pytest pytest-html pytest_html_merger pytest-reportlog pytest-rerunfailures cloudpickle portpicker matplotlib absl-py flatbuffers hypothesis


