FROM keunhobyeon/magpie2025:latest
WORKDIR /workspace

COPY ./ResShift .
COPY ./inference_sample/weights ./weights
