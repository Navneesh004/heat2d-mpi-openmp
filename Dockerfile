# Reproducible Linux build/run environment (useful on Windows/macOS).
#   docker build -t heat2d .
#   docker run --rm -it -v "$PWD:/work" heat2d
# perf usually does NOT work inside containers; profile on native Linux.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake openmpi-bin libopenmpi-dev \
        python3 python3-numpy python3-matplotlib python3-pandas \
        util-linux procps \
    && rm -rf /var/lib/apt/lists/*

# OpenMPI refuses to run as root by default; use a normal user.
RUN useradd -m hpc
USER hpc
WORKDIR /work
CMD ["bash"]
