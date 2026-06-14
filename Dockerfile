# Lambda-compatible image. Single stage keeps it simple; the base already has a
# slim Python. Large model artifacts load from S3 to /tmp on cold start, so they
# are not baked in here.
FROM public.ecr.aws/lambda/python:3.11

COPY pyproject.toml ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir \
    fastapi uvicorn mangum pydantic onnxruntime pillow exifread requests \
    transformers torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ ${LAMBDA_TASK_ROOT}/src/

CMD ["src.api.handler.handler"]
