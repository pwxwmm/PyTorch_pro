Author / 作者: mmwei3 (韦蒙蒙)
Date / 日期: 2025-09-13
Weather / 天气: Sunny / 晴

## PyTorch AIOps Interview Demos (Logs + Metrics)

中文 | English

#### TL;DR / 摘要
- 生产脱敏 Demo，展示 PyTorch 与 AIOps、监控自动化的结合，用于SRE人员接触AIOPS尝试。
- 指标预测：predict_linear 基线 + LSTM 增强，Pushgateway 写回，双确认/残差告警降噪。
- 日志与自定义指标：CMDB 自动发现 + public_exporter 采集 GPU 特定指标（如 Mellanox 闪断）。
- 训练失败 RCA：多标签文本分类，集成科研平台与告警通知，降低工单与沟通成本。
- 博客地址：https://mmwei.blog.csdn.net/article/details/151657065?fromshare=blogdetail&sharetype=blogdetail&sharerId=151657065&sharerefer=PC&sharesource=qq_28513801&sharefrom=from_link

<details>
<summary>中文背景（展开/收起）</summary>

这三个实例demo是近期从生产环境中抽象出来的，为了避免泄露敏感信息，已对数据、配置与细节做了脱敏与抽象，方便大家快速了解和学习 PyTorch 框架如何结合 AIOps 和监控自动化. 这里简单说下为什么做这个，做这个能给自己的业务和团队带来什么好处：我们团队在2019-2023年基本上都是平台类的运维业务，比如对象存储、块存储、文件存储，云计算平台(openstack、k8s）、云管平台等平台类业务运维，要求可用性7*24，稳定性也要求很严格，集团内大多数生产业务都在上面，比如医疗、教育、汽车三大核心部门业务，初期采用的是纯开源prometheus全套方案，在自己开发一套监控平台，结合自建自演进的自动化运维平台的webhook方式,来实现告警自处理(alertmanegr->监控平台->运维平台(webhook)->自动化运维平台(celery+ansible-playbook))，不过这个方式仅仅能适合一些比较固化类的场景，大多数场景还是无法覆盖，并且随便时间的推移、业务的增加，单纯的告警和自动化已经无法满足我们的运维保障，因为我理解，告警是很被动的，让sre知道是什么组件或者什么工具崩了，不能提供服务了，等知道的时候，告警已经产生了，已经是过去时，并且难以做到SRE真正的应急保障1-5-10-30原则。而如果想提前发现，则需要采用预测类的手段，简单理解为过去将来时，(这点和英语很像)，所以这时候就需要引入新的预测或者观测技术来超前保障了，这里也使用过prometheus本身的方案，比如大家熟悉的expr: predict_linear(quota_group_use_ratio{clu_name="train28",group="cog8"}[6h], 43200) > 98， 这里就需要要求你的数据必须是线性，或者近似线性的，从建模能力方面他不如LSTM，因为很多时候，运维场景几乎不是线性的，就拿这个最基础的配额业务告警，有时候数据进行删除或者增长，很难保证能做线性回归，因为存在突变性、滞后性，比如用户大量的删除数据，这样预测就不准了，特征输入方面，predict_linea也比较单一，基本上都是时间序列，而LSTM可以多个变量，比如(CPU/GPU/IO/log)等联合建模，非常适合作因果和特征融合。所以我这边采用了LSTM将预测后的结果通过pushgatway的方式，在回推给prometheus，当作一个push方式的metrics，再结合叠加predict_linear(quota_group_use_ratio{clu_name="train28",group="cog8"}[6h], 43200) > 98，组成predict_linear(quota_group_use_ratio{group="cog8"}[6h], 43200) > 0.98 and on(group) ai_forecast_quota_group_use_ratio{group="cog8", horizon="12h"} > 0.98这种方式，就相对更佳准确，鲁棒性更强。当前由于人力和时间问题，该LSTM的落地规模还是较小，但是认为这是一个非常现实且能帮助运维降低告警噪音的一个非常有效的手段。因为当前想让大模型去根据业务告警执行对应的变更类命令还是很难控制的，不过做一些基础的查询类白名单还是可以的，比如ceph集群的某个osd故障了，或者出现大量的req_block，他可以执行ceph -s类查询命令，排序，把req_block最多的osd打印出来，询问是否执行systemctl stop ceph-osd@xx.service.这样也能减轻运维人员的重复操作，因为都是7*24h保障，偶尔A/B岗都无法联系的时候，或者电脑不在身边的时候，也能通过该小模型，做快速恢复类操作，这样就将上面早期固化类的webhook，调整为可以控制的，询问的，等待主人下发命令的运维智能体了，但是我认为还不能是AIOPS，因为还需要人控制，还需要不断优化，梳理自己的生产运维需求。 前段时间也做了一个基于YI-9B的开源模型，做了一个监控智能小助手，比如可以问他：“今天rabbitmq的告警怎么这么多？是啥原因？” 他会告诉你答案。这个实现方式也非常简单，训练推理后的结果通过tornado的handler_api 进行接收，运维通过使用字节的arcodesign框架开发的一个简单的web交互UI，数据存在mongodb中。但是依然需要完善，需要不断的解析线上生产环境的json数据喂给他。

第二个是日志检测，这里简单说下，因为随着集团内工作调动，后续支持智算场景下的大模型SRE可观测建设，和自动化类建设。这里使用的GPU卡有些需要的metrics指标是无法直接通过xxx_exporter获取的，这里我通过使用golang开发了一个public_exporter(https://github.com/pwxwmm/public_exporter#)，他的功能很简单，就是通过自动化平台，根据cmdb中的指定卡型和标签自动发现哪些节点需要部署自定义exporter组件，就自动下发该组件和对应的脚本仓库中的shell或者python脚本。以及默认的脚本定时执行时间，将脚本输出的内容当作metrics通过public_exporter的5535端口暴露给prometheus，这样就能监控特定的场景了，比如A800，A100，H20这类GPU卡的mlx网卡闪断，因为闪断大概率会影响到用户的训练任务，如果不及时发现和解决，会影响失败任务的排查和训练，造成算力资源的浪费。

第三个是训练失败任务分析，因为很多的研究员用户在提训练任务(通过k8s调度，以pod方式)，总是会遇到奇奇怪怪、模莫名其妙的问题，有些是新手，比如模型参数配置问题、镜像问题、代码问题、语法问题、异常抛出问题、GPU/NPU问题、闪断问题、存储配额问题、训练OOM问题、机器逻辑掉卡、物理掉卡问题。做了根因分析，其实就是训练入门的文本分类，通过numpy和pandas对数据做清洗在喂给模型，让模型不断训练和推理，通过flaskAPi方式提供给科研平台，任务失败后，触发该模型分析，分析ELK中的对应的json类的日志，将结果返回给用户，也同时提醒通知的方式返回给运维，让对应的运维进行关注，用户侧也能减少服务台失败任务工单咨询，降低沟通。当前因为数据集的问题和时间问题准确率还需要继续提高。

</details>

<details>
<summary>English Background (click to expand)</summary>

These three demo projects are abstracted from recent production work. To avoid disclosing sensitive information, data, configurations, and implementation details have been sanitized and simplified. The goal is to help readers quickly understand how PyTorch integrates with AIOps and monitoring automation for demo and interview purposes.

Background and motivation: From 2019 to 2023 our team primarily operated platform services (object/block/file storage, OpenStack, K8s, cloud management) under 24×7 availability and strict stability requirements. Our early approach used the Prometheus stack plus an in‑house monitoring platform and an automation platform connected via webhooks (Alertmanager → Monitoring Platform → Ops Platform (Webhook) → Automation Platform (Celery + Ansible Playbook)). This worked for fixed scenarios but, as the business grew, reactive alerts could not meet SRE’s 1‑5‑10‑30 response principle, so we introduced predictive/observability capabilities.

Metric forecasting practice: we use PromQL predict_linear as a baseline. For complex, non‑linear and seasonal patterns, we add an LSTM that can fuse multiple signals (CPU/GPU/IO/logs, etc.). We push forecasts back via Pushgateway as ai_forecast_* metrics and combine them with predict_linear (dual‑confirmation or residual alerts) to reduce noise and improve lead time. The current rollout is small due to bandwidth, but it has proved effective for reducing alert noise.

Beyond runbooks with LLMs: we are cautious about letting an LLM execute change actions. We start with a whitelist of read‑only diagnostics (e.g., for Ceph OSD issues, run and rank ceph -s/req_block, then ask for confirmation before actions such as systemctl stop ceph‑osd@xx.service). This light‑weight “ops agent” reduces toil in 24×7 shifts but is not full AIOps. We also built a Yi‑9B‑based assistant: results are received via a Tornado handler API, shown in a simple ArcoDesign web UI, and stored in MongoDB; it still needs continuous ingestion of production JSON to improve.

Second: log observability. After moving to AI compute SRE observability and automation, some GPU metrics are not available from standard exporters. We built a Go‑based public_exporter (https://github.com/pwxwmm/public_exporter#) that auto‑discovers nodes via CMDB, deploys custom exporters and shell/Python scripts with default schedules, and exposes metrics on port 5535 for Prometheus. This monitors scenarios such as Mellanox link flaps on A800/A100/H20 GPUs, which can cause training task failures and wasted compute if not handled promptly.

Third: training‑failure root‑cause analysis. Many researchers launching jobs (K8s pods) encounter a variety of issues: parameter/config mistakes, image/env problems, code/syntax errors, exceptions, GPU/NPU errors, link flaps, storage quota, OOM, logical/physical GPU failures. We built an RCA service as multi‑label text classification. Data are cleaned with NumPy/Pandas and fed to the model; a Flask API integrates with the research platform. On failure it analyzes the corresponding ELK JSON logs, returns results to users, and notifies SRE, reducing tickets and communication. Accuracy is improving as datasets grow.

</details>

### 内容概览 Overview
- **日志异常检测 Log Anomaly Detection**: BERT 微调二分类，输入一行日志 -> 输出 正常/异常。HuggingFace + FastAPI。
- **指标时间序列预测 Metric Forecast**: LSTM 预测下一个时间点的指标值（CPU/GPU/延迟等）。PyTorch + FastAPI。

### 环境依赖 Requirements
```bash
pip install -r requirements.txt
```

---

## 1) 指标预测 Metric Forecast (LSTM)
路径 Path: `metric-predict-demo/`

- 模型 Model: `LSTMForecast` (`nn.LSTM` + `Linear`)
- 数据 Data: CSV `data/metrics.csv` (若不存在自动生成合成数据)
- 训练 Train:
```bash
cd metric-predict-demo
python train.py --epochs 10 --seq_len 30
```
- 推理 Predict (命令行 CLI):
```bash
python predict.py --data_csv data/metrics.csv
```
- API 服务 API Server:
```bash
uvicorn api:app --reload --port 8001
# POST /predict
# body: {"series": [0.1,0.2,..., 30 points]}
```

应用示例 Use case:
- 最近 30 分钟 CPU 利用率 -> 预测下一分钟值；若 > 90% 则提前告警。

---

## 2) 日志异常检测 Log Anomaly Detection (BERT)
路径 Path: `log-anomaly-demo/`

- 模型 Model: `bert-base-chinese` + 分类头 (2 类)
- 数据 Data: `data/train.txt`, `data/val.txt` (制表符分隔: label\ttext)
- 训练 Train:
```bash
cd log-anomaly-demo
python train.py --epochs 3
```
- 推理 Predict (脚本):
```bash
python predict.py
```
- API 服务 API Server:
```bash
uvicorn api:app --reload --port 8002
# POST /predict
# body: {"text": "2025-09-10 ERROR ..."}
```

---

## 部署与集成 Deployment & Integration
- **FastAPI** 提供 HTTP 接口，可对接 Prometheus/Alertmanager、ELK、Ansible。
- 可打包为 **Docker**，部署到 **K8s**；GPU 环境可启用 `torch.cuda.is_available()`。

### 模型保存/加载 Model Save/Load
- 指标模型: `metric-predict-demo/saved_model/forecast.pt` + `metadata.json` (包含标准化参数)。
- 日志模型: `log-anomaly-demo/saved_model/` (HuggingFace `save_pretrained`).

### GPU 加速 GPU
- 训练/推理自动检测 CUDA；可用 `DataParallel` 在多卡上并行（示例未开启以简化）。

---

---

## 数据格式 Data Format
- Metrics CSV:
```
timestamp,value
1694505600,0.35
1694505660,0.42
...
```
- Logs TSV (`label\ttext`):
```
0\t2025-09-10 12:30:05 INFO Nginx request 200 OK
1\t2025-09-10 12:32:00 CRITICAL Nginx worker crashed with exit code 137
```

---

## 快速演示 Quick Demo
```bash
# Metrics
cd metric-predict-demo && python train.py && python predict.py

# Logs
cd ../log-anomaly-demo && python train.py && python predict.py
```

---

## 目录结构 Structure
```
metric-predict-demo/
  data/
  model.py
  train.py
  predict.py
  api.py
log-anomaly-demo/
  data/
  model.py
  train.py
  predict.py
  api.py
requirements.txt
```

---

## 3) 训练失败日志根因分析 Train Failure RCA (Multi-label BERT)
路径 Path: `train-failure-rca-demo/`

- 目标 Goal: 根据训练失败日志，输出可能的根因标签（可多选）与建议动作。
- 标签示例 Labels: OOM, CUDA_Driver_Mismatch, CUDA_OutOfMemory, DataLoader_Stuck, Disk_Full, Permission_Denied, Network_Timeout, Model_Code_Error
- 数据 Data: `data/train.tsv`, `data/val.tsv` (格式: `labels,labels\ttext`)
- 训练 Train:
```bash
cd train-failure-rca-demo
python train.py --epochs 3
```
- 推理 Predict:
```bash
python predict.py
```
- API 服务 API Server:
```bash
uvicorn api:app --reload --port 8003
# POST /predict
# body: {"text": "2025-09-10 ERROR CUDA out of memory ...", "threshold": 0.5}
```

说明 Notes:
- 多标签问题（multi-label），采用 `BCEWithLogits`（HuggingFace 自动处理）。
- 可根据置信度阈值筛选多个根因，并返回建议处理动作。

---

## 背景总结 Background Summary

- **核心动机 Motivation**
  - 从“被动告警”转向“预测 + 可观测”，支撑 SRE 1-5-10-30 应急原则与 7x24 保障。
  - 固化的 Webhook 自处理覆盖有限，复杂场景需要数据驱动的预测与模式识别。

- **指标预测 Metric Forecasting（基线 + 增强）**
  - 基线：PromQL `predict_linear`，零训练、成本低，适合平稳近线性场景。
  - 增强：LSTM 学习非线性/季节性并融合多指标（CPU/GPU/IO/网络/日志等）。
  - 工程：通过 Pushgateway 写回 `ai_forecast_*`，与 `predict_linear` 做双确认/残差告警，降噪与提升提前量。

- **日志与自定义指标 Logs & Custom Metrics**
  - 结合 CMDB 自动发现/下发 `public_exporter` 与脚本，暴露特定卡型/场景指标（如 Mellanox 闪断）。
  - 对大模型训练稳定性相关信号（GPU、网络、存储等）进行观测闭环。

- **训练失败根因分析 Training Failure RCA（多标签）**
  - 目标：对失败日志输出一个或多个根因标签（如 OOM、驱动不匹配、DataLoader 卡死、磁盘满、权限、网络、代码错误）。
  - 方式：多标签文本分类（BERT + BCEWithLogits），经 API 集成至科研平台与告警通知，减少工单与沟通成本；精度随数据积累持续提升。

- **工程落地 Integration**
  - 服务化：FastAPI + Docker/K8s；与 Prometheus/Alertmanager、ELK、Ansible 对接。
  - 运行时：自动检测 GPU；根据需要扩展 DataParallel/混合精度/梯度累积等能力。

- **预期价值 Expected Outcomes**
  - 降低告警噪声与误报，提升告警“提前量”和定位速度。
  - 训练失败自动归因，缩短排障时间，降低重复性人工处理。
  - 架构可演进：从单指标线性基线，平滑升级到多信号非线性与闭环联动。

---

## CI/CD + Ansible 部署指南 (Docker + K8s/VM)

### 目标
- 将三个 API (`metric-predict-demo/api.py`, `log-anomaly-demo/api.py`, `train-failure-rca-demo/api.py`) 以容器方式构建与发布。
- 通过 GitHub Actions 构建镜像并推送至镜像仓库（GitHub Container Registry 或 Harbor）。
- 使用 Ansible 在多台主机上以 systemd 或 Docker 方式一键部署；或以 K8s Deployment 方式上线。

### 示例 Dockerfile（通用模板）
将以下 `Dockerfile` 放到各 Demo 目录内（例如 `metric-predict-demo/Dockerfile`）。
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY ../../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
ENV PORT=8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
```

构建示例：
```bash
cd metric-predict-demo
docker build -t ghcr.io/<org>/metric-forecast:latest .
```

### GitHub Actions（构建并推送镜像）
放置于 `.github/workflows/build.yml`
```yaml
name: build-and-push
on:
  push:
    branches: [ main ]
jobs:
  metric:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Login GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build & Push metric
        uses: docker/build-push-action@v5
        with:
          context: ./metric-predict-demo
          push: true
          tags: ghcr.io/${{ github.repository }}/metric-forecast:latest
  logs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: ./log-anomaly-demo
          push: true
          tags: ghcr.io/${{ github.repository }}/log-anomaly:latest
  rca:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: ./train-failure-rca-demo
          push: true
          tags: ghcr.io/${{ github.repository }}/train-failure-rca:latest
```

### Ansible 部署（Docker 方式）
`inventories/hosts.ini`
```ini
[aiops_api]
10.0.0.11 ansible_user=root
10.0.0.12 ansible_user=root
```

`playbooks/deploy_docker.yml`
```yaml
- hosts: aiops_api
  become: true
  tasks:
    - name: Ensure docker installed
      package:
        name: docker.io
        state: present

    - name: Pull images
      community.docker.docker_image:
        name: "{{ item }}"
        source: pull
      loop:
        - ghcr.io/your/repo/metric-forecast:latest
        - ghcr.io/your/repo/log-anomaly:latest
        - ghcr.io/your/repo/train-failure-rca:latest

    - name: Run metric-forecast
      community.docker.docker_container:
        name: metric-forecast
        image: ghcr.io/your/repo/metric-forecast:latest
        ports:
          - "8001:8000"
        restart_policy: always

    - name: Run log-anomaly
      community.docker.docker_container:
        name: log-anomaly
        image: ghcr.io/your/repo/log-anomaly:latest
        ports:
          - "8002:8000"
        restart_policy: always

    - name: Run train-failure-rca
      community.docker.docker_container:
        name: train-failure-rca
        image: ghcr.io/your/repo/train-failure-rca:latest
        ports:
          - "8003:8000"
        restart_policy: always
```
执行：
```bash
ansible-playbook -i inventories/hosts.ini playbooks/deploy_docker.yml
```

### Ansible 部署（systemd + venv 方式）
适用于无 Docker 的内网主机。
`playbooks/deploy_systemd.yml`
```yaml
- hosts: aiops_api
  become: true
  vars:
    app_dir: /opt/aiops/metric
    venv: /opt/aiops/metric/.venv
  tasks:
    - name: Create dirs
      file:
        path: "{{ item }}"
        state: directory
      loop:
        - "{{ app_dir }}"

    - name: Sync app code
      synchronize:
        src: ../metric-predict-demo/
        dest: "{{ app_dir }}"
        delete: yes

    - name: Ensure venv
      command: python3 -m venv {{ venv }} creates={{ venv }}/bin/python

    - name: Install deps
      command: "{{ venv }}/bin/pip install -r {{ app_dir }}/../../requirements.txt"

    - name: systemd unit
      copy:
        dest: /etc/systemd/system/metric-forecast.service
        content: |
          [Unit]
          Description=Metric Forecast API
          After=network.target

          [Service]
          WorkingDirectory={{ app_dir }}
          ExecStart={{ venv }}/bin/uvicorn api:app --host 0.0.0.0 --port 8001
          Restart=always

          [Install]
          WantedBy=multi-user.target

    - name: Reload & start
      systemd:
        daemon_reload: yes
        name: metric-forecast
        state: restarted
        enabled: yes
```

### K8s 部署（可选）
`k8s/metric-forecast.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: metric-forecast
spec:
  replicas: 1
  selector:
    matchLabels: { app: metric-forecast }
  template:
    metadata:
      labels: { app: metric-forecast }
    spec:
      containers:
        - name: api
          image: ghcr.io/your/repo/metric-forecast:latest
          ports:
            - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: metric-forecast
spec:
  selector:
    app: metric-forecast
  ports:
    - port: 8001
      targetPort: 8000
```

### CI/CD 触发建议
- main 分支 push → 构建三张镜像 → 成功后自动执行 Ansible Playbook（可用 GitHub Actions 的 SSH/Ansible Action 或自建 Runner）。
- 标记版本 tag（如 v0.1.0）→ 镜像带 tag 推送 → 生产环境灰度发布。

### 运行与回滚
- 健康检查：为三路由 `/predict` 做 200/超时监控；接口保护加上简单 auth/token。
- 回滚：Ansible 变量化镜像 tag，快速切回上一个稳定版本；K8s 使用 `rollout undo`。

说明：以上文件路径仅示例，可按你的目录结构调整；镜像仓库地址替换为你自己的 `ghcr.io/<org>/<repo>` 或 Harbor。
