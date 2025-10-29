# Dockerfile

我们提供了一个 [docker/Dockerfile](../../../docker/Dockerfile)，用于构建可运行 vLLM 的 OpenAI 兼容服务器的镜像。
关于如何使用 Docker 部署的更多信息，可以参考 [这里](../../deployment/docker.md)。

下面是多阶段 Dockerfile 的可视化展示。构建流程图包含以下节点：

- 所有构建阶段
- 默认构建目标（用灰色高亮显示）
- 外部镜像（用虚线边框表示）

构建流程图中的连接线表示：

- `FROM ...` 依赖关系（实线并带有实体箭头）

- `COPY --from=...` 依赖关系（虚线并带有空心箭头）

- `RUN --mount=(.\*)from=...` 依赖关系（点线并带有空心菱形箭头）

  > <figure markdown="span">
  >   ![](../../assets/contributing/dockerfile-stages-dependency.png){ align="center" alt="query" width="100%" }
  > </figure>
  >
  > 制作工具：<https://github.com/patrickhoefler/dockerfilegraph>
  >
  > 生成构建流程图的命令（请确保在 vLLM 仓库的 **根目录** 下运行，此目录下需有 dockerfile）：
  >
  > ```bash
  > dockerfilegraph \
  >   -o png \
  >   --legend \
  >   --dpi 200 \
  >   --max-label-length 50 \
  >   --filename docker/Dockerfile
  > ```
  >
  > 如果你想直接用 docker 镜像运行，也可以这样操作：
  >
  > ```bash
  > docker run \
  >    --rm \
  >    --user "$(id -u):$(id -g)" \
  >    --workdir /workspace \
  >    --volume "$(pwd)":/workspace \
  >    ghcr.io/patrickhoefler/dockerfilegraph:alpine \
  >    --output png \
  >    --dpi 200 \
  >    --max-label-length 50 \
  >    --filename docker/Dockerfile \
  >    --legend
  > ```
  >
  > （如果你需要针对其他文件生成流程图，只需为 `--filename` 参数传入其他文件路径即可。）