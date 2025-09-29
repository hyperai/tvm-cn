---

title: Docker 镜像

---


我们提供了 Docker 实用脚本帮助开发者搭建开发环境，这些脚本也能辅助运行 TVM 的演示和教程。如需使用 CUDA，需提前安装 [docker](https://docs.docker.com/engine/installation/) 和 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/)。


通过获取 TVM 源码发行版或克隆 GitHub 仓库以获取辅助脚本：

```plain
git clone --recursive https://github.com/apache/tvm tvm
```


可通过以下命令启动 Docker 镜像：

```plain
/path/to/tvm/docker/bash.sh <image-name>
```


其中镜像名称可以是本地构建的 Docker 镜像（例如构建完成后可使用的 `tvm.ci_cpu`）


该辅助脚本会执行以下操作：
* 将当前目录挂载至 `/workspace`
* 切换至 `bash.sh` 脚本调用者的用户身份（确保可读写主机系统）
* 在 Linux 下使用主机网络，在 macOS 下使用桥接网络并暴露 8888 端口（因 macOS 不支持 host 网络驱动，此举可支持 `jupyter notebook` 运行）


启动 Jupyter notebook 后输入：

```plain
jupyter notebook
```


在 macOS 上启动时若出现 `OSError: [Errno 99] Cannot assign requested address` 错误，可通过以下命令修改绑定 IP：

```plain
jupyter notebook --ip=0.0.0.0
```


请注意，在 macOS 上因 `bash.sh` 使用 Docker 桥接网络，Jupyter 的运行地址会显示为 `http://{container_hostname}:8888/?token=...`，在浏览器访问时需将 `container_hostname` 替换为 `localhost`。


## Docker 源码


如需构建自定义 Docker 镜像，请参考 [Docker 源码目录](https://github.com/apache/tvm/tree/main/docker)。


运行以下命令构建镜像：

```plain
/path/to/tvm/docker/build.sh <image-name>
```


非官方的第三方预构建镜像可访问 [https://hub.docker.com/r/tlcpack/](https://hub.docker.com/r/tlcpack/)，这些镜像仅用于测试，不属于 ASF 正式发布版本。


