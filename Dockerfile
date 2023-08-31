# Start fresh from a pure OS
FROM fedora

# Fedora 36’s default Python is 3.10. But R3 tested Python is 3.9.
# We’ll use Fedora’s «alternatives» framework to make 3.9 the system’s default
# Python.
# Then, we’ll use Python’s core ensurepip module to get pip installed. Then
# we’ll this pip to install all the rest.
#
# If you can use OS default Python, install procedure is different. See
# https://github.com/loft-br/robson_avm/blob/9d68917e937236c050a83aa64b53a2af435e4c60/Dockerfile
#
# Avi Alkalay <avi@unix.sh>
# 2022-08-03


# DataX
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true


# Use ARG to pass variables used ONLY ON IMAGE BUILD TIME, not container (app) execution time.
# Container execution environment (and secrets) must not go in Dockerfiles because they
# leave traces, which is bad for security. App secrets and other configurations should
# go in the .env file.
#
# See: https://vsupalov.com/docker-arg-env-variable-guide/#arg-and-env-availability

ARG USER=xingu


# Include path where pip and poetry install executables
ENV USER $USER
ENV PATH "$PATH:/home/$USER/.local/bin"
ENV PYTHONPATH "$PYTHONPATH:/home/$USER/robson_avm"



# We need superuser privileges for some of the following commands
USER root

# Install all required software.
# Prefer OS packages for binary high performance libs, such as numpy or PostgreSQL.
# Install some OS tools for practical purposes
RUN    dnf update -y \
    && dnf install -y findutils curl rsync sqlite git openblas-devel \
        python3-pip python3-wheel poetry python3-fsspec \
        python3-numpy python \
        python3-fiona python3-gdal python3-greenlet python3-frozenlist python3-grpcio \
        python3-kiwisolver python3-lz4 python3-aiohttp python3-matplotlib \
        python3-multidict python3-markupsafe python3-pygit2 python3-scikit-learn

# - Set Python 3.9 as default
# - Make Python 3.9 functional
# - Create low level user and do everything else as non-root
# RUN    alternatives --install /usr/bin/python python /usr/bin/python3   1 \
#     && alternatives --install /usr/bin/python python /usr/bin/python3.9 2 \
#     && alternatives --auto python \
#     && useradd -u 2000 -m $USER
RUN useradd -u 2000 -m $USER

# Switch to a lower end user, which is the one that will run the application
USER $USER

WORKDIR /home/$USER


# Install app dependencies.
# Disable poetry's virtualenvs: usefull in multi-project laptops, not production environments
# Use plain pip to install required modules in user space, not root.
# COPY pyproject.toml ./
# RUN    python -m ensurepip \
#     && python -m pip install -U pip wheel poetry \
#     && python -m poetry config virtualenvs.create false \
#     && python -m poetry export -f requirements.txt --without-hashes --output requirements.txt \
#     && python -m pip install -r requirements.txt --user
RUN python -m pip list -v
