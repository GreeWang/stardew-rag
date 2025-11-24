# RAG系统比较实验报告

生成时间: 2025-11-24T22:27:27.756414

## 检索性能比较

| 实验配置 | 检索器类型 | Recall@1 | Recall@3 | Recall@5 | MRR |
|----------|------------|----------|----------|----------|-----|
| m3e_base_standard | dense | 0.8400 | 0.9400 | 0.9600 | 0.8917 |
| m3e_base_detailed | dense | 0.8400 | 0.9400 | 0.9600 | 0.8917 |
| m3e_base_stardew | dense | 0.8400 | 0.9400 | 0.9600 | 0.8917 |
| bge_small_standard | dense | 0.8600 | 0.9400 | 0.9600 | 0.9007 |
| bm25_standard | sparse | 0.7800 | 0.9000 | 0.9200 | 0.8373 |
| bm25_detailed | sparse | 0.7800 | 0.9000 | 0.9200 | 0.8373 |

## 答案质量比较

| 实验配置 | 检索器类型 | 忠实度 | 相关性 | 完整性 | F1 | 整体质量 |
|----------|------------|---------|---------|---------|----|----------|
| m3e_base_standard | dense | 0.3133 | 0.9667 | 0.6391 | 0.5609 | 0.6397 |
| m3e_base_detailed | dense | 0.3733 | 1.0000 | 0.9778 | 0.4033 | 0.7837 |
| m3e_base_stardew | dense | 0.4067 | 0.9000 | 0.9501 | 0.2627 | 0.7523 |
| bge_small_standard | dense | 0.3067 | 0.9333 | 0.6962 | 0.5669 | 0.6454 |
| bm25_standard | sparse | 0.3200 | 0.9333 | 0.7094 | 0.5727 | 0.6542 |
| bm25_detailed | sparse | 0.4200 | 1.0000 | 0.9430 | 0.3685 | 0.7877 |


## 详细分析

### 最佳配置推荐

- **最佳检索性能**: bge_small_standard (平均召回率: 0.9200)
- **最佳答案质量**: bm25_detailed (整体质量: 0.7877)

### 组件影响分析

#### 检索器类型比较

- **dense**: 平均召回率 0.9150, 平均质量 0.7053
- **sparse**: 平均召回率 0.8667, 平均质量 0.7210

#### 嵌入模型比较（密集检索器）

- **moka-ai/m3e-base**: 平均召回率 0.9133, 平均质量 0.7252
- **BAAI/bge-small-zh**: 平均召回率 0.9200, 平均质量 0.6454

#### 提示模板比较

- **standard**: 忠实度 0.3133, 相关性 0.9444, 完整性 0.6816, F1 0.5668, 整体 0.6464
- **detailed**: 忠实度 0.3967, 相关性 1.0000, 完整性 0.9604, F1 0.3859, 整体 0.7857
- **stardew_specific**: 忠实度 0.4067, 相关性 0.9000, 完整性 0.9501, F1 0.2627, 整体 0.7523

### 实验配置详情

#### m3e_base_standard

- 检索器类型: dense
- 嵌入模型: moka-ai/m3e-base
- 提示模板: standard

#### m3e_base_detailed

- 检索器类型: dense
- 嵌入模型: moka-ai/m3e-base
- 提示模板: detailed

#### m3e_base_stardew

- 检索器类型: dense
- 嵌入模型: moka-ai/m3e-base
- 提示模板: stardew_specific

#### bge_small_standard

- 检索器类型: dense
- 嵌入模型: BAAI/bge-small-zh
- 提示模板: standard

#### bm25_standard

- 检索器类型: sparse
- 提示模板: standard

#### bm25_detailed

- 检索器类型: sparse
- 提示模板: detailed

