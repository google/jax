#!/usr/bin/env bash
#==============================================================================
#
# setup.rocm.sh: Prepare the ROCM installation on the container.
# Usage: setup.rocm.sh <ROCM_VERSION>
set -x

# Add the ROCm package repo location
ROCM_VERSION=$1 # e.g. 5.7.0
ROCM_PATH=${ROCM_PATH:-/opt/rocm-${ROCM_VERSION}}
ROCM_DEB_REPO_HOME=https://repo.radeon.com/rocm/apt/
ROCM_BUILD_NAME=ubuntu
ROCM_BUILD_NUM=main

# Adjust the ROCM repo location
# Intial release don't have the trialing '.0'
# For example ROCM 5.7.0 is at https://repo.radeon.com/rocm/apt/5.7/
if [ ${ROCM_VERSION##*[^0-9]} -eq '0' ]; then
        ROCM_VERS=${ROCM_VERSION%.*}
else
        ROCM_VERS=$ROCM_VERSION
fi
ROCM_DEB_REPO=${ROCM_DEB_REPO_HOME}${ROCM_VERS}/

if [ ! -f "/${CUSTOM_INSTALL}" ]; then
    # Add rocm repository
    chmod 1777 /tmp
    DEBIAN_FRONTEND=noninteractive apt-get --allow-unauthenticated update
    DEBIAN_FRONTEND=noninteractive apt install -y wget software-properties-common
    DEBIAN_FRONTEND=noninteractive apt-get clean all
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -;
    if [[ $ROCM_DEB_REPO == https://repo.radeon.com/rocm/*  ]] ; then \
      echo "deb [arch=amd64] $ROCM_DEB_REPO $ROCM_BUILD_NAME $ROCM_BUILD_NUM" > /etc/apt/sources.list.d/rocm.list; \
    else \
      echo "deb [arch=amd64 trusted=yes] $ROCM_DEB_REPO $ROCM_BUILD_NAME $ROCM_BUILD_NUM" > /etc/apt/sources.list.d/rocm.list ; \
    fi
    #Install rocm and other packages
    apt-get update --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        software-properties-common \
        clang-6.0 \
        clang-format-6.0 \
        curl \
        g++-multilib \
        git \
        vim \
        libnuma-dev \
        virtualenv \
        python3-pip \
        pciutils \
        python-is-python3 \
        libffi-dev \
        libssl-dev \
        build-essential \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev curl \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        wget \
        rocm-dev \
        rocm-libs \
        miopen-hip \
        miopen-hip-dev \
        rocblas \
        rocblas-dev \
        rocsolver-dev \
        rocrand-dev \
        rocfft-dev \
        hipfft-dev \
        hipblas-dev \
        rocprim-dev \
        hipcub-dev \
        rccl-dev \
        hipsparse-dev \
        hipsolver-dev \
        wget && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*


else
    bash "/${CUSTOM_INSTALL}"
fi

echo $ROCM_VERSION
echo $ROCM_REPO
echo $ROCM_PATH
echo $GPU_DEVICE_TARGETS

# Ensure the ROCm target list is set up
GPU_DEVICE_TARGETS=${GPU_DEVICE_TARGETS:-"gfx900 gfx906 gfx908 gfx90a gfx940 gfx941 gfx942 gfx1030 gfx1100"}
printf '%s\n' ${GPU_DEVICE_TARGETS} | tee -a "$ROCM_PATH/bin/target.lst"
touch "${ROCM_PATH}/.info/version"
