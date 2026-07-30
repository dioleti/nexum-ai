# 🌐 Nexum AI — Agentic Intelligence Framework

Nexum AI é um framework completo para desenvolvimento de sistemas de Inteligência Artificial agêntica, modular,
extensível e orientada a produção.

Ele unifica:

- Multicloud Storage
- Document Processing

Tudo em uma arquitetura coerente, elegante e poderosa.

---

# ✨ Visão Geral

**Nexum AI** foi projetado para:

🔹 Processar documentos em escala
PDF, imagens, CSV, tabelas, OCR, streams binários — tudo padronizado.\
🔹 Conectar-se a qualquer storage
S3, Azure, GCS, SMB, Local — com autenticação automática.\
🔹 Ser modular e extensível
Cada parte é plugável, substituível e expansível.

---

# 🧩 Arquitetura

A arquitetura do **Nexum AI** é dividida em módulos:

```commandline
nexum/
    common/
        storage/
    multicloud/
        storage/
            loader/
            client/
        api.py
    file/
        parser/
        reader/
        storage/
            client/
            loader/
        models.py
```

Cada módulo é independente, mas todos se integram perfeitamente.

---

# 🌐 Módulo Multicloud

O módulo multicloud fornece acesso unificado a qualquer storage:

Providers suportados

- S3
- Azure Blob Storage
- Google Cloud Storage
- SMB
- Local Files

### API vibe user

```python
from nexum.multicloud import read, stream, exists, uri

data = read("s3://bucket/key.pdf")
```

### Autenticação automática

- AWS: IAM, profiles, env vars
- Azure: DefaultAzureCredential, Managed Identity
- GCP: ADC, metadata server
- SMB: env vars

---

# 📚 Módulo Document Processing

O módulo de documentos transforma bytes em estrutura:

## Readers

- PDFReader
- ImageReader
- CSVReader

## Modelo `Document`

Padronizado para RAG, NLP, LLMs e pipelines.\
Autoconversível para langchain Document

```python
import nexum.document.models

document = nexum.document.models.Document(content="sometext", metadata={})
langchain_converted = document.to_langchain_document()
```

## Parsers

- OCR
- tabelas
- imagens
- heurísticas

---

# 🧱 Extensibilidade

**Nexum AI** foi projetado para ser:

### ✔ Modular

Cada módulo é independente.

### ✔ Plugável

Substitua qualquer parte.

### ✔ Escalável

Funciona em produção.

### ✔ Padronizado

Modelos consistentes em toda a biblioteca.

---

# 📦 Instalação

```bash
uv pip install nexum-ai
```

Extras:

```bash
uv pip install nexum-ai[s3]
uv pip install nexum-ai[azure]
uv pip install nexum-ai[gcs]
uv pip install nexum-ai[smb]
uv pip install nexum-ai[ocr]
uv pip install nexum-ai[fastapi]
uv pip install nexum-ai[mcp]
```

---

# 🧪 Testes

```bash
uv pip install -e ".[dev]"
uv run pytest -q
```

---

# 📄 Licença

MIT License
Copyright © 2026 Dioleti®