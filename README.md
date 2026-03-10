# 🧠 Sistema de Triagem de Retinopatia Diabética com IA
**Faculdade Donaduzzi — Curso de Inteligência Artificial — Projeto Integrador II — 2026**

---

## 👥 Integrantes
| Nome |
|------|
| André Thiago Malher |
| Débora Bus Cristaldo |
| Érica Souza dos Santos |
| Felipe Fernandes Clivati |
| Leticia Victoria Castellani Garbellini |
| Samara Ruiz Silva |

---

## 📋 Sobre o Projeto

Sistema de triagem automatizada para **Retinopatia Diabética (RD)** baseado em *Deep Learning*, desenvolvido para apoiar a Atenção Primária do SUS no município de **Toledo-PR**.

O sistema classifica imagens de fundo de olho em **5 estágios de gravidade**, permitindo priorização de casos críticos no sistema de regulação **CARE-GSUS**, com custo operacional de apenas **3%** em relação a equipamentos tradicionais.

---

## 🏗️ Arquitetura e Pipeline

```
Imagem (smartphone + lente 20D)
        ↓
   CLAHE (realce vascular)
        ↓
   EfficientNet-B4
   (Transfer Learning ImageNet)
        ↓
   Classificação em 5 classes
        ↓
   GradCAM (explicabilidade)
```

---

## 🩺 Classes de Diagnóstico

| Classe | Descrição | Risco |
|--------|-----------|-------|
| `No_DR` | Retina saudável | ✅ Normal |
| `Mild` | Microaneurismas iniciais | 🟡 Baixo |
| `Moderate` | Hemorragias presentes | 🟠 Moderado |
| `Severe` | Bloqueio de vasos | 🔴 Alto |
| `Proliferate_DR` | Neovascularização — risco de cegueira | 🚨 Crítico |

---

## ⚙️ Técnicas Utilizadas

| Técnica | Descrição |
|---------|-----------|
| **EfficientNet-B4** | Arquitetura principal via `timm` — 380×380 px |
| **Transfer Learning** | Pesos pré-treinados no ImageNet |
| **Freeze + Fine-tuning** | Epochs 1-3 backbone congelado, depois fine-tuning completo |
| **CLAHE** | Realce de contraste vascular no espaço LAB (clipLimit=2.0) |
| **Data Augmentation** | Flip, rotação ±180°, brilho, blur — via Albumentations |
| **Class Weight** | Correção do desbalanceamento (Proliferate peso ~9.9×) |
| **Dropout 0.4** | Reduz oscilação de acurácia na camada final |
| **AMP** | Mixed Precision — ~2× mais rápido na GPU |
| **Early Stopping** | Paciência = 5 epochs |
| **Checkpoint frequente** | Salva a cada 300 batches — nunca perde progresso |
| **GradCAM** | Mapa de ativação — mostra onde a IA olhou |

---

## 📁 Estrutura do Repositório

```
retinopatia-ia/
├── TREINO_V2_B4.py                ← Treino principal EfficientNet-B4 (use este)
├── COLAR_NO_COLAB.py              ← Treino V1 EfficientNet-B0 (prototipagem)
├── DEMONSTRACAO_VALIDACAO.py      ← Validação + GradCAM + Matriz de Confusão
└── README.md
```

> ⚠️ Os arquivos `.pth` (modelo treinado) e o dataset **não estão no repositório** — são grandes demais para o GitHub.
> Acesse a pasta completa no Google Drive pelo link abaixo.

---

## 📦 Google Drive — Dataset, Modelos e Checkpoints

📂 **[Acessar pasta RETINOPATIA no Google Drive](https://drive.google.com/drive/folders/1mJV1MflA0leaRcj6Urhjqzd-W30Pp4rM?usp=drive_link)**

Estrutura completa da pasta `RETINOPATIA` no Drive:

```
RETINOPATIA/
├── No_DR/                         ← 25.817 imagens
├── Mild/                          ← 2.444 imagens
├── Moderate/                      ← 5.292 imagens
├── Severe/                        ← 873 imagens
├── Proliferate_DR/                ← 709 imagens
├── checkpoints_retina/
│   ├── b0_checkpoint.pth          ← checkpoint EfficientNet-B0
│   └── b4_checkpoint.pth          ← checkpoint EfficientNet-B4
├── retinopatia_model_final.pth    ← melhor modelo B0 salvo
└── retinopatia_b4_final.pth       ← melhor modelo B4 salvo
```

---

## 🚀 Como Rodar

### Pré-requisitos
- Conta Google com acesso ao Google Colab
- Acesso à pasta do Drive pelo link acima

### Treino (EfficientNet-B4 — recomendado)
1. Abra o Google Colab
2. Ative a GPU: `Ambiente de execução → Alterar tipo de execução → GPU (T4)`
3. Cole o conteúdo de `TREINO_V2_B4.py` em uma célula e rode
4. O modelo é salvo automaticamente em `RETINOPATIA/retinopatia_b4_final.pth`

### Validação e Demonstração
1. Certifique-se que `retinopatia_b4_final.pth` está no Drive
2. Cole o conteúdo de `DEMONSTRACAO_VALIDACAO.py` em uma célula e rode
3. Serão gerados automaticamente:
   - Accuracy geral e sensibilidade por classe
   - Matriz de Confusão
   - GradCAM dos erros críticos

---

## 📊 Dataset e Resultados (Ciclo 1 — B0)

| Info | Valor |
|------|-------|
| Fonte | Kaggle — Diabetic Retinopathy 224x224 |
| Total de imagens | 35.135 |
| Split | 80% treino / 20% validação |
| Resolução B0 | 224×224 px |
| Resolução B4 | 380×380 px |
| Melhor Val Accuracy (B0) | 55,26% — epoch 4 |
| GPU utilizada | Tesla T4 — 15,6 GB |

---

## 📚 Referências

- ALVES, D. O. *Desempenho de Retinografia Portátil com IA no Rastreio de Retinopatia Diabética na Atenção Primária*. UFRGS, 2024.
- OLIVEIRA, L. E. S. et al. *Diagnóstico da RD por IA por meio de smartphone*. Revista Brasileira de Oftalmologia, 2024.
- SILVA, J. C. et al. *Aplicação da IA no diagnóstico e prevenção da RD*. Acervo Saúde, 2025.
- BRASIL. Ministério da Saúde. VIGITEL Brasil 2023. Brasília, 2024.
- TOLEDO (PR). PPA 2026/2029. Toledo: Prefeitura Municipal, 2025.

---

## 🔒 Ética e LGPD

O projeto implementa **anonimização local** dos dados biométricos oculares, em conformidade com a **Lei Geral de Proteção de Dados (LGPD)**, garantindo privacidade dos pacientes e segurança jurídica na aplicação clínica.

---

*Desenvolvido na Faculdade Donaduzzi — Toledo, PR — 2026*
