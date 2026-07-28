# strad32 서빙용 vLLM 파생 이미지
#
# 목적
#   업스트림 vLLM 이미지는 USER 를 선언하지 않아 컨테이너가 root 로 돈다.
#   docker 기본 설정에는 user namespace 매핑이 없으므로 컨테이너의 uid 0 은
#   호스트의 uid 0 과 같다. 그 결과 바인드 마운트(HF 캐시 등)에 만들어지는
#   파일이 전부 root 소유가 되어, 이후 팀 계정이 캐시를 쓰거나 지우지 못한다.
#
#   런타임에 --user / -e USER / -e HOME 을 매번 붙여 우회할 수도 있으나,
#   비-root uid 가 컨테이너의 /etc/passwd 에 없으면 이름 조회가 필요한 코드가
#   깨진다 (torch import -> getpass.getuser() -> pwd.getpwuid() -> KeyError).
#   사용자를 이미지에 등록하면 그 부류가 통째로 사라진다.
#
# 기반 이미지 선택
#   vllm/vllm-openai 를 쓴다. vllm/vllm-omni 도 같은 커밋의 공식 빌드이지만
#   기동 시 "NVFP4 W4A4 weight_scale NaN-clamp" 패치를 전역 적용하므로,
#   NVFP4 정확도를 평가하는 단계에서 정밀도 요인과 패치 요인이 섞인다.
#
# 빌드
#   docker build -t local-vlm/vllm:v0.24.0-dmt docker/
#
#   다른 계정용으로 만들 때는 build arg 로 지정한다 (docs/05 배정표의 uid).
#   docker build -t local-vlm/vllm:v0.24.0-vpt \
#     --build-arg UNAME=vpt --build-arg UID=1004 --build-arg GID=1004 docker/
#
# 실행
#   ENTRYPOINT 는 업스트림의 ["vllm","serve"] 를 그대로 상속하므로
#   이미지 뒤에 모델명부터 넘긴다. HF_HOME 은 /cache 로 고정되어 있으므로
#   호스트의 캐시 디렉토리를 반드시 마운트할 것.
#
#   docker run -d --name vlm-r0 \
#     --gpus '"device=GPU-<uuid>"' \
#     --cpuset-cpus=0-15,32-47 --cpuset-mems=0 \
#     --memory=64g --memory-swap=64g --shm-size=16g \
#     -v /data01/dmt/hf-cache:/cache \
#     -p 8001:8000 \
#     local-vlm/vllm:v0.24.0-dmt <model> <옵션>

ARG BASE=vllm/vllm-openai:v0.24.0
FROM ${BASE}

ARG UNAME=dmt
ARG UID=1003
ARG GID=1003

RUN groupadd -g ${GID} ${UNAME} \
 && useradd -u ${UID} -g ${GID} -m -s /bin/bash ${UNAME} \
 && mkdir -p /cache \
 && chown -R ${UID}:${GID} /home/${UNAME} /cache

USER ${UNAME}

ENV HOME=/home/${UNAME} \
    HF_HOME=/cache

# 컨테이너 *내부* 서비스 포트를 문서화한다. 포트를 여는 것이 아니며,
# 호스트 노출은 실행 시 -p <호스트>:8000 / compose ports: 가 결정한다.
# vLLM 의 기본 포트이며, --port 로 바꾸면 이 값과 어긋나므로 바꾸지 않는다.
EXPOSE 8000

# 기동 완료 판정. 모델 로드 + CUDA graph 캡처에 수 분이 걸리므로
# start-period 를 넉넉히 두어 그 사이의 실패를 세지 않는다.
#   docker ps 의 (starting)/(healthy) 표시, compose 의
#   depends_on: condition: service_healthy 가 이 값을 본다.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
  CMD python3 -c "import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://localhost:8000/v1/models',timeout=5).status==200 else 1)"
