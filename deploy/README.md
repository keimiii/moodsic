# Deployment commands

Push the backend image

`docker buildx build --platform linux/amd64 -f Dockerfile.backend -t keimii/moodsic-be:latest --push .`

Push the frontend image

`docker buildx build --platform linux/amd64 -f Dockerfile.frontend -t keimii/moodsic-fe:latest --push .`