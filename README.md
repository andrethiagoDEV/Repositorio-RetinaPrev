# 🩺 Sistema Inteligente de Triagem de Retinopatia Diabética
### Faculdade Donaduzzi — Curso de Inteligência Artificial — 2026

> Classificação automática de grau de Retinopatia Diabética em imagens de fundo de olho (fundoscopia) usando Deep Learning com **ResNet-50**.

---

## 👥 Autores

| Nome |
|---|
| André Thiago Malher |
| Débora Bus Cristaldo |
| Érica Souza dos Santos |
| Felipe Fernandes Clivati |
| Leticia Victoria Castellani Garbellini |
| Samara Ruiz Silva |

---

## 📋 Sobre o Projeto

Sistema de triagem automatizada para Retinopatia Diabética voltado à **Atenção Primária do SUS**, com foco na redução das filas de espera de até 3 anos para consultas oftalmológicas na região de Toledo-PR.

O modelo classifica imagens de retina em **5 graus de severidade**:

| Grau | Classe | Risco |
|---|---|---|
| 0 | No DR (Sem retinopatia) | ✅ Normal |
| 1 | Mild (Leve) | 🟡 Baixo |
| 2 | Moderate (Moderada) | 🟠 Moderado |
| 3 | Severe (Severa) | 🔴 Alto |
| 4 | Proliferative (Proliferativa) | 🚨 Crítico |

---

## 🏆 Resultados — ResNet-50

| Métrica | Valor |
|---|---|
| **Acurácia** | **88,2%** |
| **F1-Macro** | **0.8918** |
| **AUC-ROC** | **0.9721** |
| **Kappa Quadrático** | **0.8918** |

### F1 por Classe

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| No DR | 0.8879 | 0.9179 | 0.9026 |
| Mild | 0.8992 | 0.8406 | 0.8689 |
| Moderate | 0.8023 | 0.7687 | 0.7852 |
| Severe | 0.9603 | 0.9667 | 0.9635 |
| Proliferative | 0.9367 | 0.9409 | 0.9388 |

> **EfficientNet-B0** foi usado apenas como modelo de prototipagem para validar o pipeline (acurácia 55,26% em 8 epochs). O modelo de produção é o **ResNet-50**.

---

## 📁 Estrutura do Repositório

```
📁 retinopatia-ia/
│
├── 📄 README.md
├── 📄 requirements.txt
│
├── 📓 PROJETO_INTEGRADOR_II_TREINO_3_ResNet50.ipynb      ← Treino completo ResNet-50
└── 📓 PROJETO_INTEGRADOR_II_INFERÊNCIA_RESNET50.ipynb    ← Inferência com gabarito CSV
```

---

## ☁️ Modelo Treinado e Dataset (Google Drive)

Todos os arquivos do projeto estão disponíveis no Google Drive:

| Conteúdo | Link |
|---|---|
| 📁 Pasta completa do projeto (modelo, checkpoints, resultados) | [⬇️ Acessar Drive](https://drive.google.com/drive/folders/1mJV1MflA0leaRcj6Urhjqzd-W30Pp4rM?usp=sharing) |

> O Drive contém:
> - `resnet50_retinopatia_final.pth` — modelo final treinado (99 MB)
> - `resnet50_checkpoint.pth` — checkpoint completo (295 MB)
> - Gráficos de treino e matriz de confusão
> - Resultados da inferência

---

## 🗂️ Estrutura de Pastas no Google Drive

```
📁 Meu Drive/
└── 📁 RETINOPATIA/
    │
    ├── 📁 DATAG/
    │   └── 📦 augmented_resized_V2.zip    ← Dataset de treino (3,46 GB)
    │
    ├── 📁 DATAP/
    │   ├── 📁 No_DR/
    │   ├── 📁 Mild/
    │   ├── 📁 Moderate/
    │   ├── 📁 Severe/
    │   ├── 📁 Proliferate_DR/
    │   └── 📄 trainLabels.csv             ← Gabarito para inferência
    │
    ├── 📁 checkpoints_resnet50/
    │   └── 🔵 resnet50_checkpoint.pth     ← Checkpoint (retomada de treino)
    │
    ├── 📁 resnet50_resultados/            ← Gráficos gerados pelo treino
    ├── 📁 inferencia_datap/               ← Resultados da inferência
    │
    └── 🔴 resnet50_retinopatia_final.pth  ← Modelo final
```

---

## 🚀 Como Usar

### Pré-requisitos

```
Python 3.10+
Google Colab (recomendado — GPU T4)
```

### Instalar dependências

```bash
pip install -r requirements.txt
```

### 1. Treinar o ResNet-50

1. Abra `PROJETO_INTEGRADOR_II_TREINO_3_ResNet50.ipynb` no Google Colab
2. Ative a GPU: `Ambiente de execução → Alterar tipo → GPU T4`
3. Execute as células em ordem — o modelo salva automaticamente no Drive a cada epoch

### 2. Rodar Inferência no DATAP

1. Certifique-se que `resnet50_retinopatia_final.pth` está no Drive (ou baixe pelo link acima)
2. Abra `PROJETO_INTEGRADOR_II_INFERÊNCIA_RESNET50.ipynb` no Google Colab
3. Execute as células — gera métricas, matriz de confusão e CSV de predições

---

## 🔧 Metodologia

### Pré-processamento
- Normalização ImageNet `[0.485, 0.456, 0.406]`
- Data Augmentation leve (dataset já augmentado): flip horizontal/vertical, brilho/contraste

### Arquitetura ResNet-50

```
ResNet-50 (pré-treinada ImageNet V2)
    └── Backbone (25.5M parâmetros)
         └── Dropout(0.4)
              └── Linear(2048 → 512)
                   └── ReLU
                        └── Dropout(0.3)
                             └── Linear(512 → 5)
```

### Estratégia de Treino

| Fase | Epochs | Backbone | LR |
|---|---|---|---|
| Warmup | 1–3 | Congelado | 3e-4 |
| Fine-tuning | 4–15 | Descongelado | 5e-5 |

- **Loss:** CrossEntropyLoss + label_smoothing=0.05 + class_weight
- **Otimizador:** AdamW + weight_decay=1e-4
- **Scheduler:** CosineAnnealingLR
- **AMP:** Mixed Precision (FP16)
- **Early Stopping:** paciência de 5 epochs

---

## 📦 Dataset

| Dataset | Fonte | Imagens | Uso |
|---|---|---|---|
| augmented_resized_V2 | [Kaggle](https://www.kaggle.com/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy) | 143.687 | Treino ResNet-50 |
| DATAP | Local | — | Inferência / Validação |

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos — **Projeto Integrador II**, Faculdade Donaduzzi, 2026.

---

## 📚 Referências

- He, K. et al. **Deep Residual Learning for Image Recognition**. CVPR, 2016.
- Alves, D. O. **Desempenho de Retinografia Portátil com IA no Rastreio de RD**. UFRGS, 2024.
- Oliveira, L. E. S. et al. **Diagnóstico da retinopatia diabética por IA via smartphone**. Rev. Bras. Oftalmologia, 2024.
