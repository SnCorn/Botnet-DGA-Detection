# hash:sha256:b6de74318a1177111641742fc5cb79843af9017df5ccbe761316f5fd7f3c4dbe
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-python3.7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    cvxopt==1.2.5 \
    datetime==4.3 \
    joblib==0.15.1 \
    matplotlib==3.2.1 \
    numpy==1.18.4 \
    qiskit==0.19.2 \
    sklearn==0.0
