# Base image
FROM ubuntu:22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    ca-certificates \
    software-properties-common \
    build-essential \
    gfortran \
    libgdal-dev \
    python3-opencv \
    libfftw3-dev \
    libboost-all-dev \
    libgmp-dev \
    libmpfr-dev \
    liblapack-dev \
    libblas-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libopencv-dev \
    unzip \
    cmake \
    pkg-config \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda init bash

# Accept Anaconda TOS
RUN conda config --set channel_priority strict && \
    conda tos accept --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --channel https://repo.anaconda.com/pkgs/r && \
    conda config --remove channels defaults || true && \
    conda config --add channels conda-forge


# Create env and install ISCE2 + MintPy
RUN conda config --set channel_priority strict && \
    conda create -c conda-forge -n insar_env -y python=3.9 isce2 mintpy c-compiler cartopy cvxopt rasterio dask>=1.0 dask-jobqueue>=0.3 defusedxml h5py lxml matplotlib numpy pip pyproj scikit-image scipy gdal cython setuptools pulp pyaps3>=0.3 pykml>=0.2 pyresample pysolid -y


# Set bash as default shell
SHELL ["/bin/bash", "-c"]

# # Clone and install ISCE2 using legacy method
# RUN git clone https://github.com/isce-framework/isce2.git /opt/isce2 && \
#     source /opt/conda/etc/profile.d/conda.sh && \
#     conda activate insar_env && \
#     cd /opt/isce2 && \
#     cd setup && \
#     ./install.sh -p /opt/isce2/install && \
#     echo 'export PATH=/opt/isce2/applications/:$PATH' >> ~/.bashrc && \
#     echo 'export PYTHONPATH=/opt/isce2/applications/:/opt/isce2/components/:$PYTHONPATH' >> ~/.bashrc


RUN git clone https://github.com/isce-framework/isce2.git /isce2 && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate insar_env && \
    cd /isce2 && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX && \
    make -j$(nproc) && \
    make install && \
    echo 'export PYTHONPATH=/opt/conda/envs/insar_env/packages/isce2/components:$PYTHONPATH' >> ~/.bashrc && \
    echo 'export PYTHONPATH=/opt/conda/envs/insar_env/packages/isce2/applications:$PYTHONPATH' >> ~/.bashrc && \
    echo 'conda activate insar_env' >> ~/.bashrc && \
    echo 'export PATH=${PATH}:${ISCE_HOME}/bin' >> ~/.bashrc && \
    echo 'export ISCE_STACK=/isce2/contrib/stack/' >> ~/.bashrc && \
    echo 'export PATH=/isce2/applications/:$PATH' >> ~/.bashrc && \
    echo 'export PYTHONPATH=${PYTHONPATH}:${ISCE_STACK}' >> ~/.bashrc && \
    echo 'export PATH=${PATH}:${ISCE_STACK}/topsStack' >> ~/.bashrc
    
    

RUN git clone https://github.com/insarlab/MiaplPy.git /MiaplPy && \
    source /opt/conda/etc/profile.d/conda.sh && \
    cd /MiaplPy && \
    conda activate insar_env && \
    conda install -c conda-forge snaphu -y && \
    python -m pip install . && \
    cd / && \
    wget --no-check-certificate  https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu-v2.0.5.tar.gz && \
    tar -xzf snaphu-v2.0.5.tar.gz && \
    cd snaphu-v2.0.5/src && \
    make && \
    echo 'export PATH=/snaphu-v2.0.5/bin:${PATH}' >> ~/.bashrc


# Activate environment on start
ENTRYPOINT ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate insar_env && exec bash"]

# Usage
# docker build -t insar_env_image .
# docker run -it --name insar_container -v /insar-data:/insar-data -w /insar-data insar_env_image
# docker start insar_container
# docker exec -it insar_container bash
