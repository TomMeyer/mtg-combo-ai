####################################################
# Stage 1: Build environment with all dependencies 
####################################################
FROM ubuntu:24.04 AS builder

# Install essential packages
RUN apt-get update && apt-get install -y \
    software-properties-common \  
    git \
    wget \
    curl \
    build-essential \
    htop \
    vim \
    ninja-build \
    cmake \ 
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
RUN pixi install --all -vv

# Create the shell-hook bash script to activate the environment
RUN pixi shell-hook -e dev > /home/appuser/shell-hook.sh

# Clean the cache to reduce the image size
RUN pixi clean cache -y

# extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /home/appuser/shell-hook.sh

#######################################################
# STAGE 2: Runtime environment with only the essentials
#######################################################
FROM ubuntu:24.04 AS runtime

# Create the same non-root user and necessary directories
RUN useradd -m -s /bin/bash appuser && \
    mkdir -p /home/appuser/mtg-ai && \
    chown -R appuser:appuser /home/appuser

# Set the working directory and copy the files from the build stage
COPY --from=builder / /

WORKDIR /home/appuser/mtg-ai

# Switch to the non-root user
USER appuser
ENV PATH="/home/appuser/.pixi/bin:$PATH"

# Expose the default JupyterLab port
EXPOSE 8889

ENTRYPOINT [ "/bin/bash", "/home/appuser/shell-hook.sh" ]
# Set JupyterLab to run on container start without root privileges
CMD ["pixi", "run", "jupyter-lab"]

#######################################################
# STAGE 3: mtg_ai_rag_webserver environment
#######################################################
FROM runtime AS mtg_ai_rag_webserver

# Expose the port for the mtg_ai_rag_webserver
EXPOSE 8000

# Set the command to run the mtg_ai_rag_webserver
CMD ["pixi", "run", "mtg_ai_rag_webserver"]

#######################################################
# STAGE 4: mtg_ai_webserver environment
#######################################################
FROM runtime AS mtg_ai_webserver

# Expose the port for the mtg_ai_webserver
EXPOSE 8080

# Set the command to run the mtg_ai_webserver
CMD ["pixi", "run", "mtg_ai_webserver"]
