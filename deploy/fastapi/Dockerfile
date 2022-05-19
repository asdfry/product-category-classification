FROM python:3.9-slim

WORKDIR /app

COPY . .

SHELL [ "/bin/bash", "-o", "pipefail", "-c" ]

# 한글 출력을 위한 패키지 설치 (데비안)
RUN pip install --upgrade pip && \
    apt-get update && apt-get install -y locales git && \
    localedef -f UTF-8 -i ko_KR ko_KR.UTF-8 && \
    cat requirements.txt | xargs -n 1 -L 1 pip install --timeout 60 --no-cache-dir

# 한글 출력을 위한 환경변수 등록
ENV LANG=ko_KR.UTF-8 \
    LANGUAGE=ko_KR.UTF-8 \
    LC_ALL=ko_KR.UTF-8 \
    PYTHONIOENCODING=utf-8

ENTRYPOINT [ "/bin/bash", "-i", "-c" ]

CMD [ "python app.py" ]
