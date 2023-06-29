FROM ubuntu:20.04
RUN apt-get update && apt-get install -y --no-install-recommends git

RUN pip install degrade[runscript] @ git+https://gitlab.com/iacl/degrade

COPY ./degrade/main.py /opt

ENTRYPOINT ["python", "/opt/main.py"]
