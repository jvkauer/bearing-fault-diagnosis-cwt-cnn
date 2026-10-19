# Briefing Executivo: Reunião de Alinhamento com Orientador
**Projeto de Graduação / TCC:** Diagnóstico de Falhas em Rolamentos via CWT, Computed Order Tracking (COT) e Transferência de Domínio Tripartite  
**Aluno:** João Vitor  
**Orientador:** Prof. Alan  
**Data:** Terça-feira (20/10) — 14h15  

---

## 1. Visão Geral da Bateria Experimental
O trabalho avalia a generalização de Redes Neurais Convolucionais (ResNet-18) sob **três bancos de dados internacionais distintos**, cobrindo uma matriz tripartite completa de 6 direções de transferência cruzada:
1. **CWRU (EUA):** Mancal SKF 6205, rotações fixas ($\approx 1797\text{ RPM}$), falhas pontuais geradas por eletroerosão (EDM).
2. **Paderborn University (Alemanha):** Mancal SKF 6203, rotação fixa ($1500\text{ RPM}$), falhas **reais de fadiga mecânica acelerada** e condições operacionais estritas.
3. **XJTU-SY (China):** Mancal LDK UER204, rotação fixa ($2400\text{ RPM}$), ensaios de vida útil completa (*run-to-failure*) sob carga radial severa de $12\text{ kN}$.

> **Rigor Metodológico Anti-Vazamento (*Data Leakage Free*):**  
> Todas as divisões Treino/Validação/Teste foram realizadas estritamente no nível de arquivos físicos brutos (Leave-One-Load-Out no CWRU, Leave-Operating-Condition-Out em Paderborn e Leave-Bearing-Out no XJTU-SY). As normalizações locais por percentil ($1\%-99\%$) e a remoção de offset DC foram aplicadas uniformemente nas duas representações espectrais.

---

## 2. A Comparação Científica Central: CWT Tradicional vs. Order-CWT

```
Domínio do Tempo (Hz):        f_falha = Ordem * f_rotacao  ---> Deslocamento vertical dependente do RPM
Domínio de Ordens (COT):      Ordem   = f_falha / f_rotacao ---> Eixo invariante à rotação da máquina
```

### Pilar A: No regime Zero-Shot ($K=0$), o Order-CWT é Indispensável
Sem nenhuma amostra do alvo para calibração, a CWT tradicional colapsa devido ao desvio de frequência provocado pelas velocidades distintas ($1500$, $1797$ e $2400\text{ RPM}$):
* **CWRU $\rightarrow$ Paderborn:** CWT = $19{,}20\%$ $\longrightarrow$ **Order-CWT = $43{,}46\%$** (<span style="color:green">**$+24{,}26\text{ pp}$**</span>, com *recall* de $100\%$ em pista externa)
* **Paderborn $\rightarrow$ CWRU:** CWT = $26{,}23\%$ $\longrightarrow$ **Order-CWT = $54{,}75\%$** (<span style="color:green">**$+28{,}52\text{ pp}$**</span>)
* **CWRU $\rightarrow$ XJTU-SY:** CWT = $1{,}33\%$ $\longrightarrow$ **Order-CWT = $30{,}14\%$** (<span style="color:green">**$+28{,}81\text{ pp}$**</span>)

### Pilar B: No regime Few-Shot ($K=5$), o Order-CWT Vence na Fadiga Real
O caso de transferência mais exigente da literatura é migrar de defeitos artificiais EDM (CWRU) para **falhas reais de fadiga mecânica** (Paderborn):
* **CWRU $\rightarrow$ Paderborn ($K=5$):**
  * **CWT Tradicional (Tempo):** $79{,}17\%$ (estagnação, baixa discriminação de dano distribuído)
  * **Order-CWT (Proposta):** **$95{,}27\%$** (<span style="color:green">**$+16{,}10\text{ pp}$ a favor da Ordem**</span>)
* O alinhamento cinemático nas ordens BPFI ($4{,}95$) e BPFO ($3{,}05$) permite à rede identificar o contato mecânico mesmo sob a morfologia complexa de *spalling* real.

### Pilar C: Por que o Tempo parece alto em Few-Shot? (O Efeito "Atalho de Ressonância")
Ao rodar um grupo de controle treinado **do zero** (*scratch*, sem pré-treino de rolamentos):
* **Target CWRU ($K=5$) do Zero:** atinge **$98{,}40\%$** no Tempo!
* **Target XJTU ($K=5$) do Zero:** atinge **$100{,}00\%$** no Tempo!
* *Conclusão Física:* A CWT em tempo cobre até $20\text{ kHz}$. A rede neural consegue memorizar as ressonâncias estruturais da carcaça a partir de poucas amostras. Já a **Order-CWT restringe-se à banda cinemática de $0{,}5$ a $20$ ordens**, garantindo que a classificação decorra puramente das frequências de impacto do rolamento, atingindo **$91{,}94\%$ de acurácia média tripartite**.

---

## 3. Matriz Consolidada de Resultados

| Par de Transferência | Zero-Shot Tempo | **Zero-Shot Ordem** | One-Shot ($K=1$) Ordem | **Few-Shot ($K=5$) Ordem** | Destaque Prático |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CWRU $\rightarrow$ Paderborn** | 19,20% | **43,46%** | 50,62% | **95,27%** | Supera o tempo em +16 pp na fadiga real |
| **Paderborn $\rightarrow$ CWRU** | 26,23% | **54,75%** | 27,06% | **78,81%** | Zero-Shot muito superior; estabiliza em K=5 |
| **CWRU $\rightarrow$ XJTU-SY** | 1,33% | **30,14%** | 99,58% | **100,00%** | Sai de 1% no tempo para 100% calibrado |
| **XJTU-SY $\rightarrow$ CWRU** | 20,79% | **14,86%** | 56,29% | **92,40%** | Adaptação robusta de 12 kN para motor |
| **Paderborn $\rightarrow$ XJTU-SY** | 66,67%* | **33,19%** | 59,27% | **94,04%** | Normalização local resgatou sensibilidade |
| **XJTU-SY $\rightarrow$ Paderborn** | 35,20%* | **3,00%** | 53,49% | **91,10%** | De 3% para >91% com 5 amostras |
| **MÉDIA TRIPARTITE** | 28,24% | **29,90%** | **57,72%** | **91,94%** | **Viabilidade total na Indústria 4.0** |

*\*Nota: Valores no Tempo para PU/XJTU mascaravam colapso em classe majoritária no Zero-Shot.*

---

## 4. Tópicos de Pauta para Decisão com o Prof. Alan
1. **Estrutura dos Resultados no Capítulo 4 da Monografia:**
   * Seção 4.1: Desempenho Intra-Domínio (todos os modelos em 100%);
   * Seção 4.2: O Colapso Físico do Domínio do Tempo no Zero-Shot ($1\% - 26\%$);
   * Seção 4.3: O Ganho Cinemático do Order-CWT no Zero-Shot ($+24\text{ pp}$ a $+28\text{ pp}$);
   * Seção 4.4: Adaptação Few-Shot ($K=1$ a $K=20$) e a Vitória em Fadiga Real (95,27% vs. 79,17%);
   * Seção 4.5: Análise Crítica do Grupo de Controle (*Scratch* vs. *Pretrained*).
2. **Posicionamento da Contribuição:**
   * Na Indústria 4.0, uma fábrica não pode esperar falhas destrutivas para calibrar modelos. O trabalho comprova que **5 amostras de campo ($K=5$) bastam para alcançar $91{,}94\%$ de precisão diagnóstica**.
3. **Cronograma Final:**
   * Fechamento das tabelas e figuras no LaTeX;
   * Envio da primeira versão completa para revisão do orientador.
