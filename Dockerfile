FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        make git build-essential gfortran \
        r-base r-base-dev r-cran-igraph \
        libcurl4-openssl-dev libssl-dev libxml2-dev libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY environment/requirements-l0.txt /tmp/requirements-l0.txt
RUN pip install --no-cache-dir -r /tmp/requirements-l0.txt \
    && pip install --no-cache-dir \
        duckdb==1.3.2 \
        jupyterlab==4.4.10 \
        pyarrow==20.0.0 \
        scikit-learn==1.7.2 \
        tqdm==4.67.1

RUN Rscript -e 'install.packages("fitHeavyTail", repos="https://cloud.r-project.org")'

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
