
FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Update package list and install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    libpq-dev \
    virtualenv \
    pandoc \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash appuser

# Create virtual environment
RUN virtualenv /venv

# Change ownership of the virtual environment to appuser
RUN chown -R appuser:appuser /venv

# Set working directory and change ownership
WORKDIR /app
RUN chown appuser:appuser /app

RUN mkdir /work
RUN chown appuser:appuser /work

# Switch to non-root user
USER appuser

# Copy requirements file
COPY --chown=appuser:appuser requirements.txt .

# Install Python packages using virtualenv pip
RUN /venv/bin/pip3 install --no-cache-dir -r requirements.txt


# Add virtualenv to PATH so commands use the virtualenv by default
ENV PATH="/venv/bin:$PATH"

# Set default command to bash
CMD ["/bin/bash"]
