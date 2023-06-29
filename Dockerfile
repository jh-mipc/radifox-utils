FROM ubuntu:20.04
RUN apt-get update && apt-get install -y --no-install-recommends git python3 python3-pip

RUN pip3 install git+https://gitlab.com/iacl/degrade

ENTRYPOINT ["apply-degrade"]
