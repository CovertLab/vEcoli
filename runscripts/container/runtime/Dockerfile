# Container image #1: wcm-runtime.
# This Dockerfile builds the runtime environment for the whole cell model.
#
# To build this image locally from the vEcoli/ project root directory:
#
#     > docker build -f cloud/docker/runtime/Dockerfile -t ${USER}-wcm-runtime .
#
# (To build using the Cloud Build service and store in the Container Registry,
# run `cloud/build.sh`.)
#
# Add option `--build-arg from=ABC` to build from a different base image "ABC"
# but DO NOT USE an alpine base since the simulation math comes out different!
# See https://pythonspeed.com/articles/alpine-docker-python/ for more reasons.
ARG from=python:3.11.10@sha256:15f10b142547000e2429615b3d314474ae7d6ac062a8dd2c9246adcee7068e55
FROM ${from}

RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc \
    && echo "alias ll='ls -l'" >> ~/.bashrc \
    && cp ~/.bashrc /

# Update and install in the same layer so it won't install from old updates.
RUN apt-get update \
    && apt-get install -y swig=4.1.0-0.2 gfortran=4:12.2.0-3 \
    llvm=1:14.0-55.7~deb12u1 cmake=3.25.1-1 nano=7.2-1+deb12u1 \
    libopenblas-dev=0.3.21+ds-4

# This gets more consistent results from openblas.
ENV OPENBLAS_NUM_THREADS=1

# Install all the pips within one Docker layer, don't cache the downloads, and
# enable Aesara to write into the data dir it gets when running as a user with
# no home dir. (That happens because DockerTask runs tasks inside the Docker
# container with the uid:gid of the user outside the container to get
# read/write/delete access to the tasks's output files.)
COPY requirements.txt /
RUN (b1="" \
    && echo "Installing pips with '$b1'" \
    && pip install --no-cache-dir --upgrade pip setuptools==73.0.1 wheel \
    && pip install --no-cache-dir numpy==1.26.4 $b1 \
    && pip install --no-cache-dir -r requirements.txt $b1 \
    && umask 000 && mkdir -p /.aesara)

CMD ["/bin/bash"]
