FROM python:3.8

MAINTAINER Mitodru Niyogi <mitodru.niyogi@gmail.com>

RUN mkdir /DataScienceCodingChallenge
COPY requirements.txt setup.py Makefile README.md /DataScienceCodingChallenge/

RUN make -C /DataScienceCodingChallenge install-reqs

WORKDIR /DataScienceCodingChallenge
