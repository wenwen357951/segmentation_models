FROM tensorflow/tensorflow:latest-gpu AS RunningTime
LABEL com.wennest.trclab.segmentation_models.author=S.W.-Chen(wenwen357951@gmail.com)
WORKDIR project

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt \
    pip cache purge