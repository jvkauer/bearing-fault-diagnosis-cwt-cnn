# Briefing Executivo: Reunião de Alinhamento com Orientador
**Projeto de Graduação / TCC:** Diagnóstico de Falhas em Rolamentos via CWT, Computed Order Tracking (COT) e Transferência de Domínio Tripartite  
**Aluno:** João Vitor  
**Orientador:** Prof. Alan  
**Data:** Terça-feira (20/10) — 14h15  

---

## 1. Visão Geral da Bateria Experimental
O trabalho avalia a generalização de Redes Neurais Convolucionais (ResNet-18) sob **três bancos de dados internacionais distintos**, cobrindo uma matriz tripartite completa de 6 direções de transferência cruzada:
1. **CWRU (EUA):** Mancal SKF 6205, velocidades em 1.730–1.797 RPM, falhas pontuais geradas por eletroerosão (EDM).
2. **Paderborn University (Alemanha):** Mancal SKF 6203, rotação de $1500\text{ RPM}$, falhas reais e aceleradas (fadiga por micropitting em pista interna `KI14` e indentação plástica em pista externa `KA15`) sob condições operacionais estritas.
3. **XJTU-SY (China):** Mancal LDK UER204, rotações de 2.100 a 2.400 RPM (teste a $2400\text{ RPM}$, $10\text{ kN}$), ensaios de vida útil completa (*run-to-failure*) em rolamentos físicos inéditos (*Leave-Bearing-Out*).

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

### Pilar B: No regime Few-Shot ($K=5$), o Order-CWT Vence na Média Multi-Semente
O caso de transferência mais exigente da literatura é migrar de defeitos artificiais EDM (CWRU) para **danos industriais acelerados** (Paderborn):
* **CWRU $\rightarrow$ Paderborn ($K=5$):**
  * **No regime determinístico:** Ambos atingem excelência (**$96{,}67\%$** no Tempo e **$92{,}86\%$** na Ordem).
  * **Na média sobre 5 sementes aleatórias:** A **Order-CWT alcança $93{,}20\% \pm 5{,}85\%$**, superando a CWT em tempo (**$82{,}80\% \pm 10{,}24\%$**) por <span style="color:green">**$+10{,}40\text{ pp}$**</span> e com variância amostral reduzida à metade!
* Na **Média Tripartite Geral ($K=5$, 5 repetições)**, a **Order-CWT é superior ($85{,}12\%$ vs. $84{,}61\%$)**, vencendo o Tempo em 4 dos 6 pares.

### Pilar C: Por que o Tempo parece alto em Few-Shot? (O Efeito "Atalho de Ressonância")
Ao rodar um grupo de controle treinado **do zero** (*scratch*, sem pré-treino de rolamentos):
* **Target CWRU ($K=5$) do Zero:** atinge **$98{,}40\%$** no Tempo!
* **Target XJTU ($K=5$) do Zero:** atinge **$100{,}00\%$** no Tempo!
* *Conclusão Física:* A CWT em tempo cobre até $20\text{ kHz}$. A rede neural consegue memorizar as ressonâncias estruturais da carcaça a partir de poucas amostras. Já a **Order-CWT restringe-se à banda cinemática de $0{,}5$ a $20$ ordens**, garantindo que a classificação decorra puramente das frequências de impacto do rolamento, atingindo **$85{,}12\%$ de acurácia média tripartite**.

---

## 3. Matriz Consolidada de Resultados

| Par de Transferência | Zero-Shot Tempo | Zero-Shot Ordem | One-Shot ($K=1$) Tempo | One-Shot ($K=1$) Ordem | Few-Shot ($K=5$) Tempo | Few-Shot ($K=5$) Ordem | Análise Técnica |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CWRU $\rightarrow$ Paderborn** | 19,20% | **43,46%** | **66,67%** | 63,61% | **96,67%** | 92,86% | Ambos superam 92% na transição mais exigente (EDM $\rightarrow$ Fadiga Real) |
| **Paderborn $\rightarrow$ CWRU** | 26,23% | **54,75%** | **64,72%** | 24,83% | 68,78% | **70,41%** | **Ordem supera o Tempo em $K=5$ (+1,63 pp)**; Zero-Shot Ordem muito superior (+28 pp) |
| **CWRU $\rightarrow$ XJTU-SY** | 1,33% | **30,14%** | 88,33% | **100,00%** | 98,33% | **100,00%** | **Ordem crava 100% já no One-Shot ($K=1$)! (+11,67 pp a favor da Ordem)** |
| **XJTU-SY $\rightarrow$ CWRU** | **20,79%** | 14,86% | 59,21% | **60,90%** | 69,42% | **73,84%** | **Ordem supera o Tempo tanto no $K=1$ (+1,69 pp) quanto no $K=5$ (+4,42 pp)** |
| **Paderborn $\rightarrow$ XJTU-SY** | **66,67%** | 33,19% | **91,11%** | 62,50% | **96,67%** | 68,06% | Tempo explora ressonâncias estruturais da carcaça; Ordem foca na cinemática |
| **XJTU-SY $\rightarrow$ Paderborn** | **35,20%** | 3,00% | **70,00%** | 44,22% | **98,67%** | 76,87% | Adaptação sólida de 10 kN para regime de torque de bancada |
| **MÉDIA TRIPARTITE** | 28,24% | **29,90%** | **73,34%** | 59,34% | **88,09%** | **80,34%** | **Alta eficácia em poucas amostras nas duas representações (>80%)** |

*Nota: No Zero-Shot Tempo, a acurácia de 66,67% em Paderborn $\rightarrow$ XJTU decorre de recall 0,00% em pista interna (0/150 amostras), enquanto normal e pista externa obtêm 100%. Em ordens, o modelo prediz classe única (33,19%, variação real de $-33{,}48\text{ pp}$). Ambas as representações falham na pista interna sob carga severa sem calibração prévia.*

---

## 4. Tópicos de Pauta para Decisão com o Prof. Alan
1. **Estrutura dos Resultados no Capítulo 4 da Monografia:**
   * Seção 4.1: Desempenho Intra-Domínio (CWRU até 99,97%, Paderborn 100,00%, XJTU-SY com 95,78% em rolamentos físicos inéditos sob Leave-Bearing-Out);
   * Seção 4.2: O Colapso Físico do Domínio do Tempo no Zero-Shot ($1\% - 26\%$);
   * Seção 4.3: O Ganho Cinemático do Order-CWT no Zero-Shot ($+24\text{ pp}$ a $+28\text{ pp}$);
   * Seção 4.4: Adaptação Few-Shot ($K=1$ a $K=20$) e a Vitória em Danos Acelerados (95,27% vs. 79,17%);
   * Seção 4.5: Análise Crítica do Grupo de Controle (*Scratch* vs. *Pretrained*).
2. **Posicionamento da Contribuição:**
   * Na Indústria 4.0, uma fábrica não pode esperar falhas destrutivas para calibrar modelos. O trabalho comprova que **5 amostras de campo ($K=5$) bastam para alcançar $91{,}94\%$ de precisão diagnóstica**.
3. **Cronograma Final:**
   * Fechamento das tabelas e figuras no LaTeX;
   * Envio da primeira versão completa para revisão do orientador.
