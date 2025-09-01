# Imagen oficial de NVIDIA con TensorFlow 2.19 y CUDA 12.x
FROM nvcr.io/nvidia/tensorflow:24.03-tf2-py3

# Instalar dependencias adicionales para Sionna
RUN apt-get update && apt-get install -y --no-install-recommends \
        llvm \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*


# Instalar Sionna y librerías
RUN pip install --upgrade pip setuptools wheel \
    && pip install \
        "sionna==1.0.2" \
        "numpy>=1.26,<2.0" \
        "scipy>=1.14.1" \
        "matplotlib>=3.10" \
        "importlib_resources>=6.4.5" \
        "pytest>=8.3.4"


WORKDIR /workspace
CMD ["/bin/bash"]
