# 使用 Nginx

本文档介绍如何启动多个 vLLM 服务容器，并通过 Nginx 作为它们之间的负载均衡器。

## 构建 Nginx 容器

本指南假设你已经克隆了 vLLM 项目，并且当前位于 vllm 的根目录下。

```bash
export vllm_root=`pwd`
```

创建一个名为 `Dockerfile.nginx` 的文件：

```dockerfile
FROM nginx:latest
RUN rm /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

构建 Nginx 容器镜像：

```bash
docker build . -f Dockerfile.nginx --tag nginx-lb
```

## 创建简单的 Nginx 配置文件

创建一个名为 `nginx_conf/nginx.conf` 的文件。你可以根据需要添加任意数量的服务器。下面的示例以两个服务器为例，如果需要扩展，只需在 `upstream backend` 里再添加一行如 `server vllmN:8000 max_fails=3 fail_timeout=10000s;`。

??? console "配置文件"

    ```console
    upstream backend {
        least_conn;
        server vllm0:8000 max_fails=3 fail_timeout=10000s;
        server vllm1:8000 max_fails=3 fail_timeout=10000s;
    }
    server {
        listen 80;
        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
    ```

## 构建 vLLM 容器

```bash
cd $vllm_root
docker build -f docker/Dockerfile . --tag vllm
```

如果你处于代理环境下，可以通过如下方式在构建镜像时传递代理参数：

```bash
cd $vllm_root
docker build \
    -f docker/Dockerfile . \
    --tag vllm \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy
```

## 创建 Docker 网络

```bash
docker network create vllm_nginx
```

## 启动 vLLM 容器

注意事项：

- 如果你的 HuggingFace 模型缓存位于其他目录，请修改下面的 `hf_cache_dir` 路径。
- 如果你还没有 HuggingFace 缓存，建议先启动 `vllm0`，等模型下载完成并且服务就绪后，再启动 `vllm1`。这样可以避免重复下载模型，提升效率。
- 下面的示例假定你使用的是 GPU。如果你使用 CPU，请去掉 `--gpus device=ID`，并在 docker run 命令里添加 `VLLM_CPU_KVCACHE_SPACE` 和 `VLLM_CPU_OMP_THREADS_BIND` 这两个环境变量。
- 如果你不想用 `Llama-2-7b-chat-hf`，可以根据需要修改模型名称。

??? console "启动命令"

    ```console
    mkdir -p ~/.cache/huggingface/hub/
    hf_cache_dir=~/.cache/huggingface/
    docker run \
        -itd \
        --ipc host \
        --network vllm_nginx \
        --gpus device=0 \
        --shm-size=10.24gb \
        -v $hf_cache_dir:/root/.cache/huggingface/ \
        -p 8081:8000 \
        --name vllm0 vllm \
        --model meta-llama/Llama-2-7b-chat-hf
    docker run \
        -itd \
        --ipc host \
        --network vllm_nginx \
        --gpus device=1 \
        --shm-size=10.24gb \
        -v $hf_cache_dir:/root/.cache/huggingface/ \
        -p 8082:8000 \
        --name vllm1 vllm \
        --model meta-llama/Llama-2-7b-chat-hf
    ```

!!! note
    如果你处于代理环境下，可以通过 `-e http_proxy=$http_proxy -e https_proxy=$https_proxy` 参数将代理设置传递给 docker run 命令。

## 启动 Nginx

```bash
docker run \
    -itd \
    -p 8000:80 \
    --network vllm_nginx \
    -v ./nginx_conf/:/etc/nginx/conf.d/ \
    --name nginx-lb nginx-lb:latest
```

## 检查 vLLM 服务是否启动完成

```bash
docker logs vllm0 | grep Uvicorn
docker logs vllm1 | grep Uvicorn
```

你应该会看到类似如下内容：

```console
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```