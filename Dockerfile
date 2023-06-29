FROM ubuntu:20.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip

RUN pip3 install git+https://gitlab.com/iacl/degrade[runscript]

COPY ./degrade/main.py /opt

ENTRYPOINT ["python3", "/opt/main.py"]
