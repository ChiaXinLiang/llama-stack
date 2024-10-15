docker image pull llamastack/llamastack-local-cpu
docker image tag llamastack/llamastack-local-cpu llamastack-local-cpu:latest
llama stack configure llamastack/llamastack-local-cpu
llama stack run local-cpu

docker run -it -p 7777:7777 -v /Users/marcus/.llama/builds/docker/local-cpu-run.yaml:/app/config.yaml -v /Users/marcus/.llama:/root/.llama  llamastack-local-cpu python -m llama_stack.distribution.server.server --yaml_config /app/config.yaml --port 7777

llama stack build --template local-ollama --name my-local-stack