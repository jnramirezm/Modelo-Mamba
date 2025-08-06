# Dockerfile para Hepatic Vessel Segmentation con UNet + Mamba
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Establecer directorio de trabajo
WORKDIR /app

# Instalar Python 3.10 + pip y dependencias del sistema
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    git wget curl build-essential \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-venv && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/local/bin/pip /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# Variables de entorno para CUDA
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Copiar requirements y wheels
COPY requirements_simple.txt .
COPY wheels/ ./wheels/

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_simple.txt && \
    pip install wheels/Normal/*.whl

# Copiar el código fuente completo
COPY src/ ./src/
COPY set_mode.py .
COPY show_stats.py .
COPY analyze_results.py .
COPY README.md .
COPY CONFIGURACION.md .
COPY VISUALIZACION.md .

# Crear directorios para resultados (datos se montan como volúmenes)
RUN mkdir -p /app/outputs /app/models /app/logs

# Variables de entorno adicionales
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV IS_TESTING=True

# Puerto para Jupyter (opcional)
EXPOSE 8888

# Usuario no-root para seguridad
RUN useradd -m -u 1000 researcher && chown -R researcher:researcher /app
USER researcher

# Comando por defecto
CMD ["python", "src/main.py"]
