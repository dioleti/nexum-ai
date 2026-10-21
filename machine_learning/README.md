## Nexum AI

### Machine Learning Library

Biblioteca de modelos e abstrações de Machine Learning do ecossistema Nexum AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Release](https://img.shields.io/badge/Release-v1.0.0-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

</div>

---

## 📋 Sobre o Projeto

Esta biblioteca fornece componentes de Machine Learning e Processamento de Linguagem Natural (NLP) para o ecossistema
Nexum AI, incluindo:

- **Abstrações de Modelos (`BaseModel`, `StatefulModel`)**: Contratos padronizados para inferência síncrona (`predict`,
  `predict_batch`), assíncrona (`apredict`, `apredict_batch`) e gerenciamento de ciclo de vida de modelos em memória
  (`load`, `unload`).
- **Detecção de Dados Pessoais em Textos (`PIIDetector`)**: Detector híbrido (expressões regulares + NER com Hugging Face
  Transformers) capaz de analisar strings de qualquer tamanho (com *chunking* automático) para identificação de dados
  pessoais (CPF, CNPJ, e-mail, telefone, nomes e localidades) — ideal para uso direto ou como *tool* em agentes (ADK).
- **Detecção de Dados Pessoais em Colunas (`PIIColumnDetector`)**: Especialização que utiliza o `PIIDetector` para
  inspecionar amostras de colunas tabulares com suporte a limiar (`threshold`).

---

## 📦 Instalação

Utilizando o `uv`:

```bash
uv add nexum-ai-machine-learning
```

Ou via `pip`:

```bash
uv pip install nexum-ai-machine-learning
```

---

## 🚀 Exemplo de Uso

### 1. Detectando PII em uma string (ex: *Tool* de Agente ADK)

```python
from nexum_ai_machine_learning.detectors.pii_detector import PIIDetector

detector = PIIDetector()

texto = "O cliente João Souza (CPF 123.456.789-00) reside em São Paulo."
resultado = detector.predict(texto)

print(resultado.is_pii)             # True
print(resultado.confidence)         # 1.0
print(resultado.detected_entities)  # {'REGEX_PATTERN': 1, 'CPF': 1, 'PER': 1, 'LOC': 1}
```

### 2. Detectando PII em colunas tabulares (`PIIColumnDetector`)

```python
from nexum_ai_machine_learning.common.models import PiiDetectionInput
from nexum_ai_machine_learning.detectors.pii_detector import PIIColumnDetector

col_detector = PIIColumnDetector()

inputs = [
    PiiDetectionInput(
        column_name="documento",
        samples=["123.456.789-00", "111.222.333-44"],
        threshold=0.5,
    ),
    PiiDetectionInput(
        column_name="cliente",
        samples=["Maria Silva", "João Souza"],
        threshold=0.5,
    ),
]

results = col_detector.predict_batch(inputs)
for col, res in zip(inputs, results):
    print(f"{col.column_name}: is_pii={res.is_pii} (confidence={res.confidence}, entities={res.detected_entities})")

col_detector.unload()
```

---

## 🛠️ Testando Local

Para rodar os testes e validações localmente, utilizamos o gerenciador de dependências `uv`.

1. Certifique-se de ter o `uv` instalado em sua máquina.
2. Instale as dependências e o ambiente virtual:

```bash
uv sync --all-extras --dev
```

3. Execute a suíte de testes:

```bash
uv run pytest
```

4. Execute os testes com relatório de cobertura:

```bash
uv run coverage run --source=nexum_ai_machine_learning -m unittest discover -s tests -v
uv run coverage report -m
```

---

## 🤝 Contribuindo

Contribuições são sempre bem-vindas! Para garantir um fluxo de desenvolvimento organizado, siga os passos abaixo:

- Faça um fork do repositório.
- Crie uma branch para a sua feature (`git checkout -b minha-feature`).
- Faça o commit das suas alterações (`git commit -m 'feat: Adiciona nova funcionalidade X'`).
- Envie para a branch principal (`git push origin minha-feature`).
- Abra um Pull Request detalhando suas alterações.

---

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---
`Copyright (c) 2026 Dioleti(R) - Fabrício Paulo`
