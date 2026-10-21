## Nexum AI

### Common Artifact library

Biblioteca de artefatos comuns do ecossistema Nexum AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Release](https://img.shields.io/badge/Release-v1.0.0-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

</div>

---

## 📋 Sobre o Projeto

Esta biblioteca centraliza utilitários, modelos e artefatos compartilhados entre os serviços do ecossistema Nexum AI,
garantindo consistência, padronização e reuso de código.

---

## 📦 Instalação com Extras

```bash
uv pip install nexum-ai-common
```

Esta biblioteca suporta recursos opcionais para integração com monitoramento e observabilidade. Você pode instalar
dependências extras (como Datadog e Elasticsearch) utilizando o `uv`:

```bash
# Instalando com suporte ao Datadog
uv add "nexum-common[datadog]"

# Instalando com suporte ao Elasticsearch
uv add "nexum-common[elastic]"

# Instalando ambos os extras simultaneamente
uv add "nexum-common[datadog,elastic]"

```

---

## 🛠️ Testando Local

Para rodar os testes localmente, utilizamos o gerenciador de dependências uv em conjunto com o pytest.

1. Certifique-se de ter o uv instalado em sua máquina.
2. Instale as dependências e o ambiente virtual:

```bash
uv sync --all-extras --dev
```

3. Execute a suíte de testes com o pytest:

```bash
uv run pytest
```

---

## 🤝 Contribuindo

Contribuições são sempre bem-vindas! Para garantir um fluxo de desenvolvimento organizado, siga os passos abaixo:

- Faça um fork do repositório.
- Crie uma branch para a sua feature (git checkout -b minha-feature).
- Faça o commit das suas alterações (git commit -m 'feat: Adiciona nova funcionalidade X').
- Envie para a branch principal (git push origin minha-feature).
- Abra um Pull Request detalhando suas alterações.

---

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---
`Copyright (c) 2026 Dioleti(R) - Fabrício Paulo`
