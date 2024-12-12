# Use uma imagem base leve com Python
FROM python:3.12.3-slim

# Instale as dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instale o Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Adicione o Poetry ao PATH
ENV PATH="/root/.local/bin:$PATH"

# Defina o diretório de trabalho no container
WORKDIR /app

# Copie os arquivos do projeto para o container
COPY pyproject.toml poetry.lock /app/

# Instale as dependências do Poetry
RUN poetry install --no-root

# Copie o restante dos arquivos do projeto
COPY . /app

# Use ENTRYPOINT to specify the base command
ENTRYPOINT ["poetry", "run", "python", "inference.py"]

# CMD will be the default arguments, which can be overridden at runtime
CMD ["--sentence", "I would like to make sure you are happy"]