# 🔥 Firebase JSON Reader — Streamlit App

## Pré-requisitos
- Python 3.9 ou superior
- Uma conta Firebase com um projeto criado

## Instalação

```bash
# Navegue até o diretório do projeto
cd caminho/para/firebase-streamlit-app

# Instale as dependências
pip install -r requirements.txt
```

## Como executar

```bash
streamlit run app.py
```

O app abrirá automaticamente em `http://localhost:8501`.

---

## Como obter as credenciais do Firebase

1. Acesse o [Firebase Console](https://console.firebase.google.com/)
2. Selecione seu projeto → ⚙️ **Configurações do projeto**
3. Aba **Contas de serviço**
4. Clique em **Gerar nova chave privada**
5. Salve o arquivo `.json` gerado
6. Faça o **upload** desse arquivo na barra lateral do app

---

## Recursos do App

| Recurso | Descrição |
|---|---|
| 🔌 Realtime Database | Lê dados em qualquer caminho (ex: `/usuarios`) |
| 🔌 Firestore | Lê todos os documentos de uma coleção |
| 📋 Tabela interativa | Com busca e filtro em tempo real |
| 🧩 JSON Bruto | Visualização formatada com syntax highlight |
| 📊 Gráficos automáticos | Barras, linha, dispersão, histograma via Plotly |
| ⬇ Exportar | Baixar dados como `.json` ou `.csv` |

---

## Estrutura de arquivos

```
firebase-streamlit-app/
├── app.py              # Aplicação principal
├── requirements.txt    # Dependências Python
├── .env.example        # Modelo de variáveis de ambiente
└── README.md           # Este arquivo
```
