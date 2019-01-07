PROJECT_NAME := grotile
SHARED := $(if $(SHARED_VOLUME), -v $(SHARED_VOLUME):$(SHARED_VOLUME),)


docker_gpu_384:
	docker build -t ${PROJECT_NAME}_gpu -f docker/Dockerfile-gpu-384.90 .

docker_gpu_375:
	docker build -t ${PROJECT_NAME}_gpu -f docker/Dockerfile-gpu-375.26 .

docker:
	docker build -t ${PROJECT_NAME}_dev -f docker/Dockerfile .

dev:
	docker rm ${PROJECT_NAME}_dev_${USER} || true
	docker run -it --name ${PROJECT_NAME}_dev_${USER} ${SHARED} -v $(CURDIR):/workspace/ -v /var/run/docker.sock:/var/run/docker.sock:ro -p 8888:8888 --entrypoint bash ${PROJECT_NAME}_dev

run_gpu:
	docker-compose -f docker/docker-compose.yml up -d grotile-gpu
	docker exec -it docker_grotile-gpu_1 bash

.PHONY: docker run docker_gpu run_gpu
