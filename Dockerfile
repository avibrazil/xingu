# Start fresh from a pure high quality OS
FROM fedora

# Fedora Linux ships an excellent and up to date Python build along with high
# quality packages of NumPy and Pandas. Different from Anaconda, it is free.
#
# Fedora also ships different versions of Python (3.11, 3.10, 3.9). Use
# its «alternatives» framework to chose a different version as commented below.
#
# Avi Alkalay <avi@unix.sh>
# 2022-08-03


# Use ARG to pass variables used ONLY ON IMAGE BUILD TIME, not container (app) execution time.
# Container execution environment (and secrets) must not go in Dockerfiles because they
# leave traces, which is bad for security. App secrets and other configurations should
# go in the .env file.
#
# See: https://vsupalov.com/docker-arg-env-variable-guide/#arg-and-env-availability

ARG USER
ARG UID

# Include path where pip and poetry install executables
ENV USER $USER
ENV UID $UID
ENV PATH "$PATH:/home/$USER/.local/bin"



# We need superuser privileges for some of the following commands
USER root

# Install all required software.
# Prefer OS packages for binary high performance libs, such as numpy or PostgreSQL.
# Install some OS tools for practical purposes
RUN    dnf update -y && useradd -u $UID -m $USER
RUN    dnf install -y findutils curl rsync sqlite git python \
        python3-pandas openblas-devel python3-pygit2 python3-scikit-learn \
        python3-tabulate python3-unidecode python3-sqlalchemy python3-pyyaml \
        ltrace strace \
        

# Uncomment this if you need a different version of Python shipped by Fedora
# - Set Python 3.9 as default
# - Make Python 3.9 functional
# RUN    alternatives --install /usr/bin/python python /usr/bin/python3   1 \
#     && alternatives --install /usr/bin/python python /usr/bin/python3.9 2 \
#     && alternatives --auto python

# Switch to a lower priority user, which is the one that will run the application
USER $USER

WORKDIR /home/$USER

# Install app dependencies.
# Disable poetry's virtualenvs: usefull in multi-project laptops, not production environments
# Use plain pip to install required modules in user space, not root.

# Uncomment this if using a different Python version shipped by Fedora
# RUN    python -m ensurepip \
#     && python -m pip install -U pip wheel poetry \
#     && python -m poetry config virtualenvs.create false \
#     && python -m poetry export -f requirements.txt --without-hashes --output requirements.txt \
#     && python -m pip install -r requirements.txt --user

# Install everything that is needed by your model, including Xingu
RUN pip install xingu catboost scikit-optimize pyathena filprofiler memory-profiler

# Build image:
#
#   cat Dockerfile | docker build --build-arg UID=$(id -u) --build-arg USER=model_trainer -t xingu -
#
#
# Run container:
#
#   docker run --mount type=bind,source="`pwd`",destination=/home/model_trainer/mymodels -t xingu /bin/sh -c "cd mymodels; xingu"