# 用 PyTorch 打造 AIOps 小体系：日志异常、指标预测与训练失败根因分析（三个可运行 Demo）

Author / 作者: mmwei3 (韦蒙蒙)
Date / 日期: 2025-09-13
Repo / 仓库: https://github.com/pwxwmm/PyTorch_pro

> 这三个实例 demo 源自生产实践的脱敏与抽象，目的是让SRE/读者能快速上手、跑通，并看到 PyTorch 在 AIOps 场景中的实际落地方式。全文涵盖：场景动机、三类 Demo 的代码结构与运行方式、与 Prometheus/ELK 的对接、以及 CI/CD + Ansible 的工程化部署建议。

---

## 目录
- 背景与动机
- Demo 1：日志异常检测（BERT 微调）
- Demo 2：指标时间序列预测（LSTM）
- Demo 3：训练失败日志根因分析（多标签 BERT）
- 与 Prometheus / ELK 的集成
- CI/CD + Ansible 部署方案
- 经验总结与扩展方向

---

## 背景与动机
平台类运维（对象/块/文件存储，OpenStack、K8s、云管等）常见挑战：
- 告警偏“事后通知”，缺乏提前量，难支撑 7x24 与 SRE 1-5-10-30 响应原则。
- 固化 Runbook 与 Webhook 自处理覆盖有限，复杂场景需要数据驱动与模式识别。
- 训练任务失败排障成本高、沟通成本高。

思路升级：
- 将“被动告警”升级为“预测 + 可观测 + 自动化”。
- 基线采用 PromQL `predict_linear`；复杂场景引入 LSTM/Transformer 融合多指标建模。
- 对日志做文本分类（异常检测、根因分析），将结果反馈至科研平台/告警系统，形成闭环。

项目地址：[`pwxwmm/PyTorch_pro`](https://github.com/pwxwmm/PyTorch_pro)

---

## Demo 1：日志异常检测（BERT 微调）
路径：`log-anomaly-demo/`

- 技术栈：PyTorch + HuggingFace Transformers + FastAPI
- 任务：二分类（正常/异常）
- 代码结构：
  - `model.py`：`bert-base-chinese` + 分类头
  - `train.py`：读取 `data/train.txt`、`data/val.txt`（TSV: label\ttext），训练与保存
  - `predict.py`：推理脚本（输入一行日志，输出“正常/异常”）
  - `api.py`：FastAPI 服务 `/predict`

快速运行：
```bash
cd log-anomaly-demo
python train.py --epochs 3
python predict.py
uvicorn api:app --reload --port 8002
```

数据样例（TSV）：
```
0\t2025-09-10 12:30:05 INFO Nginx request 200 OK
1\t2025-09-10 12:32:00 CRITICAL Nginx worker crashed with exit code 137
```

可扩展点：
- 多类别标签（类型化异常：网络/存储/权限等）
- 大模型蒸馏/量化、加速部署（ONNX/TensorRT）

---

## Demo 2：指标时间序列预测（LSTM）
路径：`metric-predict-demo/`

- 技术栈：PyTorch（LSTM）+ FastAPI
- 任务：输入过去 `seq_len` 个点，预测下一个点值
- 代码结构：
  - `model.py`：`LSTMForecast`
  - `train.py`：训练脚本（自动生成合成数据；标准化参数写入 `metadata.json`）
  - `predict.py`：加载 `forecast.pt` 与 `metadata.json`，做单步预测
  - `api.py`：FastAPI 服务 `/predict`

快速运行：
```bash
cd metric-predict-demo
python train.py --epochs 10 --seq_len 30
python predict.py
uvicorn api:app --reload --port 8001
```

工程建议：
- 用 Pushgateway 写回 `ai_forecast_*` 指标，与 `predict_linear` 做“双确认/残差告警”降噪。
- 支持多步预测（滚动）、不确定性（分位数/区间）。

---

## Demo 3：训练失败日志根因分析（多标签 BERT）
路径：`train-failure-rca-demo/`

- 技术栈：PyTorch + HuggingFace（多标签）+ FastAPI
- 任务：对训练失败日志做多标签根因识别（可多选）
- 标签集（示例）：OOM、CUDA_Driver_Mismatch、CUDA_OutOfMemory、DataLoader_Stuck、Disk_Full、Permission_Denied、Network_Timeout、Model_Code_Error
- 代码结构：
  - `labels.json`：标签与建议动作映射
  - `model.py`：`BertForSequenceClassification(problem_type="multi_label_classification")`
  - `train.py`：读取 `data/train.tsv`、`data/val.tsv`（labels,labels\ttext），训练与保存 `label_space.json`
  - `predict.py`：多标签推理 + 阈值过滤 + 建议动作解释
  - `api.py`：FastAPI 服务 `/predict`

快速运行：
```bash
cd train-failure-rca-demo
python train.py --epochs 3
python predict.py
uvicorn api:app --reload --port 8003
```

集成建议：
- 失败任务触发时（K8s Job/Pod Failed 事件）→ 调用 `/predict` → 将根因与建议回填到科研平台与通知系统。

---

## 与 Prometheus / ELK 的集成
- Prometheus：
  - 指标预测 API 作为“预测器”，结果写回 Pushgateway，命名 `ai_forecast_*`；PromQL 融合：
  ```promql
  predict_linear(quota_group_use_ratio{group="cog8"}[6h], 43200) > 0.98
  and on(group) ai_forecast_quota_group_use_ratio{group="cog8", horizon="12h"} > 0.98
  ```
  - 残差告警：`actual - forecast` 超阈值时报警。
- ELK：
  - 将日志清洗成 `label\ttext` 训练集；线上把错误堆栈/关键信息聚合为短文本，调用日志/根因 API。

---

## CI/CD + Ansible 部署方案（摘要）
- Dockerfile 模板（各 demo 通用）：
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY ../../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
ENV PORT=8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
```
- GitHub Actions：
  - 三个 Job 分别构建并推送三张镜像（见仓库 `README.md` 示例）
- Ansible：
  - Docker 方式：拉取镜像，启动 `metric-forecast`/`log-anomaly`/`train-failure-rca` 三容器，分别映射 `8001/8002/8003`。
  - systemd + venv 方式：无 Docker 内网环境的替代方案。
- K8s：
  - 提供 Deployment + Service 样例，直接上线到集群。

详细文件与脚本在仓库 `README.md` 的“CI/CD + Ansible 部署指南”章节。

---

## 经验总结与扩展方向
- 组合策略：`predict_linear` 作为轻量基线，LSTM/Transformer 作为复杂模式增强，双确认或残差告警降噪。
- 模型演进：
  - 指标侧：支持多步预测、异常检测（AutoEncoder/Isolation Forest/TCN 等）。
  - 日志侧：从二分类到细粒度根因标签，多任务学习（分类 + 建议生成）。
- 工程与运维：增加健康检查、速率限制、鉴权；自动回滚；模型/数据漂移监控与再训练流水线。
- 团队协作：以 API/指标为契约，逐步把“预测/分析结果”纳入现有告警、工单、自动化闭环。

---

## 参考与仓库
- GitHub 项目：[`pwxwmm/PyTorch_pro`](https://github.com/pwxwmm/PyTorch_pro)
- Prometheus 基线预测：`predict_linear(...)`
- HuggingFace Transformers 文档：`https://huggingface.co/docs/transformers`
