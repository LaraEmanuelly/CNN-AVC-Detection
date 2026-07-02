# Classificador de AVC 

## Identificação

- Nome do estudante: Lara Emanuelly da Silva Santos
- Disciplina: Aprendizado Profundo
- Modelo implementado: CNN
- Dataset utilizado: Brain Stroke Dataset Binary
- Link do dataset: https://huggingface.co/datasets/BTX24/tekno21-brain-stroke-dataset-binary

## Descrição do problema
O Acidente Vascular Cerebral (AVC) é uma das principais causas de morte e incapacidade funcional no mundo. O diagnóstico rápido por imagem, como a Tomografia Computadorizada (TC) cerebral, é fundamental para que a equipe médica inicie a intervenção a tempo de salvar o parênquima cerebral. Este projeto aborda o uso de Redes Neurais Convolucionais (CNN) para automatizar a triagem de exames de imagem, realizando a classificação binária de TCs entre cérebros saudáveis e cérebros com sinais de AVC, funcionando como uma ferramenta de suporte à decisão clínica em cenários de emergência.

## Dataset
O conjunto de dados utilizado é o *Brain Stroke Dataset Binary*, obtido diretamente do repositório público Hugging Face. Ele é composto por imagens de tomografias computadorizadas axiais do cérebro humano.
- **Classes:** O problema possui duas classes mutuamente exclusivas: `0` para Paciente Saudável e `1` para Paciente com AVC.
- **Formato dos dados:** Imagens digitais que são carregadas e transformadas em tensores normalizados com dimensões estandardizadas de 224×224 pixels e 3 canais de cor.
- **Divisão dos dados:** Os dados foram divididos de maneira estratificada e controlada através de semente aleatória (`seed=42`) nas seguintes proporções:
  - **Treinamento (70%):** Utilizado para o ajuste dos pesos e aprendizado dos filtros da rede.
  - **Validação (15%):** Utilizado para monitoramento do erro durante as épocas e acionamento do critério de parada precoce (*Early Stopping*).
  - **Teste (15%):** Conjunto isolado para a aferição final e imparcial da capacidade de generalização do modelo.

```python
  transformacao_treino = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

## Modelo
Foi construída uma arquitetura convolucional profunda (CNN) customizada em PyTorch. A rede possui uma estrutura extratora composta por 4 blocos convolucionais consecutivos, onde cada bloco aplica filtros espaciais, normalização em lote para acelerar a convergência e subamostragem espacial. A etapa classificadora conta com uma camada de achatamento, uma camada densa intermediária com exclusão aleatória de neurônios para mitigar o sobreajuste e uma saída linear binária. A estrutura detalhada da classe é apresentada a seguir:

```text
RedeCNN(
  (convolucoes): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
    (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (12): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classificador): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=25088, out_features=128, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=128, out_features=1, bias=True)
  )
)
```

## Ambiente
- Linguagem: Python
- Bibliotecas: torch, torchvision, scikit-learn (sklearn), datasets, matplotlib, seaborn, numpy
- Versão do Python: 3.12+
- Uso de GPU: sim

## Principais resultados
Métricas do conjunto de testes:
- Acurácia: 94.58%
- Precisão (macro): 0.9432
- Recall (macro): 0.9405
- F1-Score (macro): 0.9418

## Como executar
```bash
pip install datasets matplotlib seaborn scikit-learn
```
