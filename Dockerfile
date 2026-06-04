FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "--version"]
