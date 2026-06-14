# Deploy to AWS Lambda (free tier, container image)

Genuinely free for demo traffic. The trade-off is cold-start latency (first
request can take several seconds). Upgrade path: provisioned concurrency or a
small EC2 instance when budget allows.

## 0. Prerequisites
- AWS CLI configured, an ECR repo, and an IAM role using `iam-policy.json`.
- Region in an env var: `export AWS_REGION=us-east-1` and `export ACCOUNT=<id>`.

## 1. Build the image
```bash
docker build -t y1-provenance .
```

## 2. Push to ECR
```bash
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
aws ecr create-repository --repository-name y1-provenance || true
docker tag y1-provenance:latest $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/y1-provenance:latest
docker push $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/y1-provenance:latest
```

## 3. Create the function from the container
```bash
aws lambda create-function \
  --function-name y1-provenance \
  --package-type Image \
  --code ImageUri=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/y1-provenance:latest \
  --role arn:aws:iam::$ACCOUNT:role/y1-provenance-role \
  --memory-size 2048 --timeout 30
```

## 4. Public URL
```bash
aws lambda create-function-url-config --function-name y1-provenance --auth-type NONE
```

## 5. Large artifacts in S3
Put the DeBERTa model and the ONNX image model in an S3 prefix. On cold start the
app downloads them to `/tmp`. The IAM role grants read on that one prefix only.

## 6. Smoke test
```bash
curl -s "$FUNCTION_URL/health"
curl -s -X POST "$FUNCTION_URL/predict/text" -H 'content-type: application/json' \
  -d '{"text":"a sentence you have never tested before"}'
```
