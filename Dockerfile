FROM nvcr.io/nvidia/tritonserver:24.07-py3

WORKDIR /workspace

RUN pip install --no-cache-dir \
    transformers \
    numpy \
    torch==2.4.0

COPY model_repository /models

EXPOSE 8000

CMD ["tritonserver", \
     "--model-repository=/models", \
     "--allow-http=true"]

