FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "--version"]