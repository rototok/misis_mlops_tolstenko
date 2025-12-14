FROM nvcr.io/nvidia/tritonserver:24.01-py3

WORKDIR /workspace

RUN pip install --no-cache-dir \
    transformers \
    numpy

COPY model_repository /models

EXPOSE 8000

CMD ["tritonserver", \
     "--model-repository=/models", \
     "--allow-http=true"]

