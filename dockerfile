####################################################
# Stage 1: Build environment with all dependencies 
####################################################
FROM continuumio/miniconda3 AS builder

# Install essential packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    htop \
    vim \
    && apt-get clean

# Create a non-root user and switch to that user
RUN useradd -m -s /bin/bash appuser && \
mkdir -p /home/appuser/workspace && \
chown -R appuser:appuser /home/appuser

# Set the working directory and copy the project files
COPY --chown=appuser:appuser . /home/appuser/mtg-ai 
WORKDIR /home/appuser/mtg-ai

# Switch to the non-root user
USER appuser

# Install Pixi package manager
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/home/appuser/.pixi/bin:$PATH"

# Install the project dependencies
RUN pixi install -vv --all

# Create the shell-hook bash script to activate the environment
RUN pixi shell-hook -e dev > /home/appuser/shell-hook.sh

# extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /home/appuser/shell-hook.sh

#######################################################
# STAGE 2: Runtime environment with only the essentials
#######################################################
FROM continuumio/miniconda3 AS runtime

# Create the same non-root user and necessary directories
RUN useradd -m -s /bin/bash appuser && \
    mkdir -p /home/appuser/mtg-ai && \
    chown -R appuser:appuser /home/appuser

# Set the working directory and copy the files from the build stage
WORKDIR /home/appuser/mtg-ai
COPY --from=builder / /
# COPY --chown=appuser:appuser --from=builder /home/appuser/mtg-ai .
# COPY --chown=appuser:appuser --from=builder /home/appuser/.pixi /home/appuser/.pixi
# COPY --chown=appuser:appuser --from=builder /home/appuser/shell-hook.sh /home/appuser/shell-hook.sh

# Switch to the non-root user
USER appuser
ENV PATH="/home/appuser/.pixi/bin:$PATH"

# Expose the default JupyterLab port
EXPOSE 8889

ENTRYPOINT [ "/bin/bash", "/home/appuser/shell-hook.sh" ]
# Set JupyterLab to run on container start without root privileges
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8889", "--no-browser"]