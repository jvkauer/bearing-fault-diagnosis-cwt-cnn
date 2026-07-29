# 🛠️ Guia Completo de Setup — bearing-fault-diagnosis-cwt-cnn

## Pré-requisitos
- ✅ Python 3.12 instalado
- ✅ Repositório criado no GitHub (com licença MIT)
- ✅ VSCode instalado

---

## ETAPA 1 — Instalar o Git (se ainda não tiver)

Abra o PowerShell e teste:
```
git --version
```
- Se aparecer a versão → Pule para a Etapa 2.
- Se der erro → Instale com o comando:
```
winget install --id Git.Git --accept-package-agreements --accept-source-agreements
```
Depois **feche e reabra** o PowerShell.

---

## ETAPA 2 — Configurar o Git (só precisa fazer UMA vez na vida)

```
git config --global user.name "Seu Nome Completo"
git config --global user.email "seu-email-do-github@email.com"
```

---

## ETAPA 3 — Clonar o Repositório

Escolha uma pasta no seu PC onde quer guardar o projeto (ex: `C:\Projetos`).

```
mkdir C:\Projetos
cd C:\Projetos
git clone https://github.com/SEU-USUARIO/bearing-fault-diagnosis-cwt-cnn.git
cd bearing-fault-diagnosis-cwt-cnn
```

> [!TIP]
> A URL do clone você pega no botão verde **"<> Code"** no seu repositório do GitHub.

---

## ETAPA 4 — Abrir no VSCode

Ainda no PowerShell, dentro da pasta do projeto:
```
code .
```
Isso abre o VSCode já na pasta correta. **A partir de agora, use o terminal integrado do VSCode** (atalho: `Ctrl + '`).

---

## ETAPA 5 — Instalar Extensões no VSCode

Na barra lateral esquerda, clique no ícone de quadradinhos (Extensões) e instale:

| Extensão | Para quê |
|---|---|
| **Python** (Microsoft) | Suporte ao Python, linting, debug |
| **Jupyter** (Microsoft) | Rodar notebooks `.ipynb` no VSCode |
| **GitLens** (opcional) | Visualizar histórico do Git de forma bonita |

---

## ETAPA 6 — Criar o Ambiente Virtual (VENV)

No terminal do VSCode (dentro da pasta do projeto):

```
python -m venv venv
```

Ativar o ambiente:
```
.\venv\Scripts\activate
```

> [!IMPORTANT]
> Você vai saber que deu certo quando aparecer `(venv)` no início da linha do terminal.
> **Toda vez** que abrir o VSCode para trabalhar no projeto, precisa ativar o venv de novo com o comando acima.

---

## ETAPA 7 — Instalar as Bibliotecas

Com o `(venv)` ativo, rode:
```
pip install numpy scipy matplotlib jupyter
```

Essas são as bibliotecas iniciais para explorar os dados:

| Biblioteca | Para quê |
|---|---|
| `numpy` | Manipulação de arrays e operações numéricas |
| `scipy` | Ler os arquivos `.mat` do CWRU |
| `matplotlib` | Plotar gráficos (sinal no tempo, FFT, etc.) |
| `jupyter` | Rodar notebooks interativos |

> [!NOTE]
> As bibliotecas de Deep Learning (`torch`, `torchvision`, etc.) vamos instalar depois, quando chegarmos na etapa de construir a CNN.

---

## ETAPA 8 — Criar a Estrutura de Pastas

No terminal do VSCode:
```
mkdir data
mkdir notebooks
mkdir src
```

A estrutura final deve ficar assim:
```
bearing-fault-diagnosis-cwt-cnn/
├── data/              ← Arquivos .mat do CWRU (NÃO vai pro GitHub!)
├── notebooks/         ← Notebooks Jupyter de exploração
├── src/               ← Scripts Python organizados (futuro)
├── venv/              ← Ambiente virtual (NÃO vai pro GitHub!)
├── .gitignore         ← Arquivo que diz o que o Git deve IGNORAR
├── LICENSE            ← Licença MIT (já veio do GitHub)
├── README.md          ← Descrição do projeto (já veio do GitHub)
└── requirements.txt   ← Lista de bibliotecas usadas
```

---

## ETAPA 9 — Criar o .gitignore

Crie um arquivo chamado `.gitignore` na raiz do projeto com o seguinte conteúdo:

```
# Ambiente virtual
venv/

# Dados (pesados demais pro GitHub)
data/

# Cache do Python
__pycache__/
*.pyc

# Checkpoints de Jupyter
.ipynb_checkpoints/

# Arquivos do sistema
.DS_Store
Thumbs.db
```

> [!CAUTION]
> Esse arquivo é **essencial**. Sem ele, você vai subir acidentalmente centenas de MB de dados `.mat` e a pasta inteira do `venv` para o GitHub.

---

## ETAPA 10 — Salvar a Lista de Bibliotecas (requirements.txt)

Com o `(venv)` ativo:
```
pip freeze > requirements.txt
```

> [!TIP]
> Rode esse comando toda vez que instalar uma biblioteca nova. Assim qualquer pessoa (ou você mesmo em outro PC) pode recriar seu ambiente com `pip install -r requirements.txt`.

---

## ETAPA 11 — Fazer o Primeiro Commit

```
git add .
git status
```

Confira se `data/` e `venv/` **NÃO aparecem** na lista (o `.gitignore` deve estar filtrando). Se estiver tudo certo:

```
git commit -m "chore: setup inicial do projeto"
git push origin main
```

> [!NOTE]
> Se o Git pedir autenticação, ele vai abrir uma janela do navegador para você fazer login no GitHub. Siga o fluxo e autorize.

---

## ETAPA 12 — Baixar os Dados do CWRU

1. Acesse: https://engineering.case.edu/bearingdatacenter
2. Baixe os arquivos `.mat` de **Normal Baseline Data** e **12k Drive End Bearing Fault Data**.
3. Jogue todos os arquivos `.mat` dentro da pasta `data/`.

---

## ETAPA 13 — Criar o Primeiro Notebook

1. Na pasta `notebooks/`, crie um arquivo chamado `01_exploracao_cwru.ipynb`.
2. Abra o arquivo no VSCode (a extensão Jupyter vai ativar automaticamente).
3. Na primeira célula, tente carregar um arquivo `.mat` e plotar o sinal!

---

## 🔁 Fluxo Diário de Trabalho

Toda vez que sentar para trabalhar no projeto:

```
cd C:\Projetos\bearing-fault-diagnosis-cwt-cnn
.\venv\Scripts\activate
code .
```

Ao terminar uma sessão de trabalho, salve seu progresso:
```
git add .
git commit -m "descricao do que voce fez"
git push origin main
```

---

## ✅ Checklist Resumido

- [ ] Git instalado e configurado
- [ ] Repositório clonado
- [ ] VSCode aberto no projeto
- [ ] Extensões instaladas (Python + Jupyter)
- [ ] Ambiente virtual criado e ativado
- [ ] Bibliotecas iniciais instaladas
- [ ] Estrutura de pastas criada
- [ ] `.gitignore` configurado
- [ ] `requirements.txt` gerado
- [ ] Primeiro commit feito e pushado
- [ ] Dados do CWRU baixados na pasta `data/`
- [ ] Primeiro notebook criado

---
---

# 📖 Guia Completo de Git — Referência Permanente

## 🔰 Conceitos Fundamentais

| Conceito | O que é |
|---|---|
| **Repository (repo)** | A pasta do seu projeto rastreada pelo Git |
| **Commit** | Um "save point" do seu código. Uma foto do estado atual |
| **Branch** | Uma "linha do tempo" paralela para desenvolver sem quebrar o código principal |
| **Merge** | Juntar duas branches (linhas do tempo) em uma só |
| **Remote (origin)** | O repositório no GitHub (a cópia na nuvem) |
| **HEAD** | O commit onde você está agora |
| **Staging Area** | A "sala de espera" — arquivos prontos para o próximo commit |

### Como o Git funciona (fluxo mental):
```
[Seus Arquivos] → git add → [Staging Area] → git commit → [Histórico Local] → git push → [GitHub]
```

---

## 📋 Comandos do Dia a Dia

### Ver o estado atual do projeto
```bash
git status
```
Mostra quais arquivos foram modificados, quais estão na staging area e quais não estão sendo rastreados.

### Adicionar arquivos para o próximo commit
```bash
# Adicionar TUDO que mudou
git add .

# Adicionar um arquivo específico
git add notebooks/01_exploracao_cwru.ipynb

# Adicionar uma pasta inteira
git add src/
```

### Fazer um commit (salvar o progresso)
```bash
git commit -m "feat: adicionar geração de escalogramas CWT"
```

### Enviar para o GitHub
```bash
git push origin main
```

### Baixar atualizações do GitHub
```bash
git pull origin main
```

### Ver o histórico de commits
```bash
# Resumido (uma linha por commit)
git log --oneline -10

# Detalhado
git log -5

# Visual (mostra branches como um gráfico)
git log --oneline --graph --all
```

### Ver o que mudou desde o último commit
```bash
# Ver quais arquivos mudaram
git diff --name-only

# Ver as mudanças linha por linha
git diff

# Ver mudanças de um arquivo específico
git diff notebooks/01_exploracao_cwru.ipynb
```

---

## 🌿 Branches (Ramificações)

Branches são a ferramenta mais poderosa do Git. Permitem que você trabalhe em uma funcionalidade nova sem arriscar quebrar o código que já funciona.

### Criar e trocar para uma branch nova
```bash
# Criar e já trocar para ela (atalho)
git checkout -b feature/cwt-preprocessing

# Equivalente moderno
git switch -c feature/cwt-preprocessing
```

### Listar todas as branches
```bash
git branch
```
A branch com o `*` é a que você está agora.

### Trocar de branch
```bash
git checkout main
# ou
git switch main
```

### Padrão de nomes para branches
```
feature/nome-da-funcionalidade   → Para funcionalidades novas
fix/descricao-do-bug             → Para correção de bugs
refactor/o-que-mudou             → Para reorganização de código
experiment/nome-do-teste         → Para experimentos (testar hiperparâmetros, etc.)
```

### Exemplos práticos para o seu TCC:
```bash
git checkout -b feature/cwt-preprocessing
git checkout -b feature/resnet-model
git checkout -b experiment/learning-rate-tuning
git checkout -b fix/dataloader-path-error
```

---

## 🔀 Merge (Juntar Branches)

Quando a funcionalidade da sua branch estiver pronta e funcionando, você junta ela de volta na `main`.

### Fluxo completo de merge:
```bash
# 1. Certifique-se que está na main
git checkout main

# 2. Atualize a main (por segurança)
git pull origin main

# 3. Junte a branch na main
git merge feature/cwt-preprocessing

# 4. Envie para o GitHub
git push origin main

# 5. (Opcional) Delete a branch que já foi mergeada
git branch -d feature/cwt-preprocessing
```

---

## ⚔️ Resolver Conflitos de Merge

Conflitos acontecem quando duas branches modificaram a **mesma linha** do **mesmo arquivo**. O Git não sabe qual versão manter e pede para você decidir.

### Como identificar um conflito:
Quando você rodar `git merge` e der conflito, o Git vai marcar o arquivo assim:

```python
def preprocessar_sinal(sinal):
<<<<<<< HEAD
    # Versão que estava na main
    sinal_normalizado = sinal / np.max(sinal)
=======
    # Versão que veio da branch
    sinal_normalizado = (sinal - np.mean(sinal)) / np.std(sinal)
>>>>>>> feature/cwt-preprocessing
    return sinal_normalizado
```

### Passo a passo para resolver:

1. **Abra o arquivo com conflito no VSCode.** O VSCode vai destacar o conflito com cores e botões:
   - `Accept Current Change` → Mantém o que estava na main
   - `Accept Incoming Change` → Usa o que veio da branch
   - `Accept Both Changes` → Mantém os dois
   - **Ou edite manualmente** → Apague os marcadores (`<<<<`, `====`, `>>>>`) e escreva a versão correta

2. **Salve o arquivo.**

3. **Adicione e faça commit:**
```bash
git add .
git commit -m "fix: resolver conflito no preprocessamento"
```

> [!TIP]
> A melhor forma de evitar conflitos é fazer branches pequenas e dar merge frequentemente. Não fique 2 semanas numa branch sem mergear!

---

## ⏪ Desfazer Coisas (Socorro!)

### Desfazer mudanças que AINDA NÃO foram commitadas
```bash
# Desfazer mudanças em um arquivo específico (volta ao último commit)
git checkout -- notebooks/01_exploracao_cwru.ipynb

# Desfazer TUDO (volta ao último commit)
git checkout -- .
```

> [!CAUTION]
> Esse comando **apaga** suas mudanças para sempre! Não tem Ctrl+Z depois disso.

### Tirar um arquivo da staging area (depois do git add)
```bash
git reset HEAD notebooks/01_exploracao_cwru.ipynb
```

### Desfazer o ÚLTIMO commit (mas manter as mudanças nos arquivos)
```bash
git reset --soft HEAD~1
```
Útil quando você fez commit com mensagem errada ou esqueceu de adicionar um arquivo.

### Desfazer o ÚLTIMO commit (e apagar as mudanças)
```bash
git reset --hard HEAD~1
```

> [!CAUTION]
> `--hard` apaga tudo! Só use se tiver certeza absoluta.

### Alterar a mensagem do último commit
```bash
git commit --amend -m "feat: mensagem corrigida"
```

### Adicionar arquivo esquecido ao último commit
```bash
git add arquivo_esquecido.py
git commit --amend --no-edit
```

---

## 🔍 Investigar o Histórico

### Quem escreveu cada linha de um arquivo (Blame)
```bash
git blame src/model.py
```

### Ver o que mudou em um commit específico
```bash
git show abc1234
```

### Procurar um texto em todos os commits
```bash
git log --all --oneline -S "learning_rate"
```

### Ver todos os commits que mexeram em um arquivo
```bash
git log --oneline -- notebooks/01_exploracao_cwru.ipynb
```

---

## 🏷️ Tags (Marcar Versões)

Tags são usadas para marcar versões importantes do projeto (ex: a versão final que você entregou na banca).

```bash
# Criar uma tag
git tag -a v1.0 -m "Versão final do TCC"

# Enviar a tag para o GitHub
git push origin v1.0

# Listar todas as tags
git tag
```

---

## 📝 Conventional Commits — Padrão de Mensagens

| Prefixo | Quando usar | Exemplo |
|---|---|---|
| `feat:` | Nova funcionalidade | `feat: adicionar geração de escalogramas CWT` |
| `fix:` | Correção de bug | `fix: corrigir erro na leitura do arquivo .mat` |
| `chore:` | Manutenção (não muda código) | `chore: atualizar requirements.txt` |
| `docs:` | Documentação | `docs: atualizar README com instruções de uso` |
| `refactor:` | Reorganizar código sem mudar funcionalidade | `refactor: separar funções de pré-processamento` |
| `test:` | Adicionar testes | `test: adicionar teste unitário para o dataloader` |
| `experiment:` | Testar hiperparâmetros/modelos | `experiment: testar ResNet50 com lr=0.001` |

### Exemplos reais para o seu TCC:
```bash
git commit -m "feat: implementar dataloader para arquivos .mat do CWRU"
git commit -m "feat: adicionar transformada CWT com wavelet de Morlet"
git commit -m "feat: implementar arquitetura ResNet para classificação"
git commit -m "fix: corrigir normalização dos escalogramas"
git commit -m "experiment: testar batch_size=32 vs batch_size=64"
git commit -m "docs: adicionar docstrings nas funções de preprocessamento"
git commit -m "chore: adicionar torch ao requirements.txt"
```

---

## 🗑️ Limpeza

### Deletar uma branch local
```bash
git branch -d nome-da-branch       # Só deleta se já foi mergeada
git branch -D nome-da-branch       # Força a deleção (cuidado!)
```

### Deletar uma branch remota (no GitHub)
```bash
git push origin --delete nome-da-branch
```

---

## 🆘 Situações de Emergência

### "Fiz push de algo errado para o GitHub!"
```bash
# Desfaz o último commit e faz push forçado
git reset --hard HEAD~1
git push --force origin main
```

> [!WARNING]
> `--force` reescreve o histórico do GitHub. Só use se você é o ÚNICO trabalhando no repositório (que é o caso do seu TCC).

### "Quero salvar meu trabalho atual sem fazer commit"
```bash
# Guardar no bolso (stash)
git stash

# Recuperar do bolso
git stash pop

# Ver o que tem guardado
git stash list
```
Útil quando você está no meio de algo e precisa trocar de branch urgente.

### "Quero voltar para um commit antigo só para ver como estava"
```bash
git checkout abc1234

# Para voltar ao presente
git checkout main
```

---

## 📊 Cheat Sheet Rápido

```
┌─────────────────────────────────────────────────────┐
│                    GIT CHEAT SHEET                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  BÁSICO                                             │
│  git status              → Ver estado atual         │
│  git add .               → Preparar tudo            │
│  git commit -m "msg"     → Salvar                   │
│  git push origin main    → Enviar pro GitHub        │
│  git pull origin main    → Baixar do GitHub         │
│  git log --oneline -10   → Ver histórico            │
│                                                     │
│  BRANCHES                                           │
│  git checkout -b nome    → Criar branch             │
│  git checkout main       → Voltar pra main          │
│  git merge nome          → Juntar branch            │
│  git branch -d nome      → Deletar branch           │
│                                                     │
│  DESFAZER                                           │
│  git checkout -- .       → Descartar mudanças       │
│  git reset --soft HEAD~1 → Desfazer commit          │
│  git stash               → Guardar no bolso         │
│  git stash pop           → Recuperar do bolso       │
│                                                     │
│  INVESTIGAR                                         │
│  git diff                → Ver mudanças             │
│  git blame arquivo       → Quem escreveu o quê      │
│  git log --graph --all   → Gráfico de branches      │
│                                                     │
└─────────────────────────────────────────────────────┘
```
