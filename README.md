# Prática 5 - Processamento de Sinais

<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/CEFET-RJ_logo.svg/320px-CEFET-RJ_logo.svg.png" width="120" alt="CEFET/RJ Logo"/>

# 🌊 Prática 5 — Processamento de Sinais I
### Transformadas de Wavelets & Sistemas CDMA

[![CEFET/RJ](https://img.shields.io/badge/CEFET%2FRJ-Engenharia%20de%20Controle%20e%20Automa%C3%A7%C3%A3o-1F3864?style=for-the-badge&logo=academia)](https://www.cefet-rj.br)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

</div>

---

## 📋 Sumário

- [Sobre a Prática](#-sobre-a-prática)
- [Exercícios](#-exercícios)
  - [Questão 1 — Hadamard & CDMA](#-questão-1--transformada-de-hadamard-e-sistemas-cdma)
  - [Questão 2 — Fundamentos de Wavelets](#-questão-2--fundamentos-das-transformadas-de-wavelets)
  - [Questão 3 — Sinal Não Estacionário](#-questão-3--análise-de-sinal-não-estacionário)
  - [Questão 4 — Denoising](#-questão-4--remoção-de-ruído-com-wavelets)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Como Executar](#-como-executar)
- [Dependências](#-dependências)
- [Equipe](#-equipe)

---

## 📖 Sobre a Prática

Esta prática explora dois grandes temas interligados do processamento moderno de sinais:

**1. Transformada de Hadamard** aplicada à base dos sistemas de comunicação **CDMA (3G)** — investigando codificação ortogonal, múltiplo acesso, robustez a ruído e recuperação de sinal.

**2. Transformadas de Wavelets** — análise multirresolução tempo-frequência, decomposição de sinais não estacionários e denoising adaptativo, comparado à filtragem clássica via DFT.

> 💡 **Por que Wavelets?** A Transformada de Fourier fornece *onde* estão as frequências, mas não *quando*. As Wavelets resolvem isso: alta resolução temporal em altas frequências, alta resolução em frequência em baixas frequências — espelhando o próprio sistema auditivo humano.

---

## 🧪 Exercícios

### 📡 Questão 1 — Transformada de Hadamard e Sistemas CDMA

**Notebook:** [`Questao_1.ipynb`](Questao_1.ipynb)

A base matemática do **3G** explorada em código. Sistemas CDMA permitem que múltiplos usuários transmitam simultaneamente no mesmo canal de frequência usando **códigos ortogonais** derivados da matriz de Hadamard.

#### O que foi feito

| Item | Descrição |
|------|-----------|
| **(a)** | Geração de sequências binárias $m_k[n] \in \{-1, +1\}$ para 3 usuários e codificação com matriz de Hadamard $H_8$. Análise do impacto espectral (*spread spectrum*). |
| **(b)** | Construção do sinal recebido $y[n] = x_1[n] + x_2[n] + x_3[n] + w[n]$ com ruído AWGN para $\sigma^2 \in \{10^{-3}, 10^{-2}, 10^{-1}, 1\}$. |
| **(c)** | Demonstração matemática da recuperação $\hat{m}_k[n]$ via ortogonalidade: $\hat{m}_k = \frac{1}{N} \mathbf{c}_k^\top \mathbf{y}$. |
| **(d)** | Estimativa de $\hat{m}_k[n]$ e curva de **BER × $\sigma^2$**. |
| **(e)** | Análise do efeito de código distorcido no receptor: $\hat{C} = C + W$. Demonstração do *error floor*. |
| **(f)** | Curva de BER com código distorcido e comparação com o cenário ideal. |

#### Resultado chave

```
Ganho de Processamento = N = 8  →  σ²_erro = σ²_canal / N
```

A ortogonalidade dos códigos Hadamard cancela perfeitamente a interferência entre usuários — **desde que o código do receptor seja exato**. Distorções no código introduzem um patamar mínimo de erro irredutível (error floor), independente da qualidade do canal.

---

### 🌊 Questão 2 — Fundamentos das Transformadas de Wavelets

**Notebook:** [`Questão2.ipynb`](Questão2.ipynb)

Resumo teórico completo sobre Wavelets e suas aplicações em processamento de sinais.

#### Conceitos abordados

- **Análise multirresolução (MRA):** banco de filtros passa-baixas/passa-altas em cascata diádica
- **CWT vs DWT:** transformada contínua e discreta, vantagens e limitações de cada uma
- **Principais famílias:** Haar · Daubechies · Biortogonais · Morlet · Mexican Hat
- **Princípio de Heisenberg adaptativo:** janelas estreitas para alta frequência, largas para baixa

#### Aplicações práticas

```
🎵 Compressão de áudio e imagens    →  JPEG 2000, codecs modernos
🏥 Sinais biomédicos                →  ECG, EEG, detecção de picos
⚙️  Diagnóstico de máquinas         →  Falhas em rolamentos e engrenagens
🌍 Geofísica                        →  Análise sísmica e estratigrafia
📡 Telecomunicações                 →  Modulação OFDM, ADSL
```

---

### 📈 Questão 3 — Análise de Sinal Não Estacionário

**Notebook:** [`Questão3.ipynb`](Questão3.ipynb)

**Wavelet utilizada:** `bior4.4` (Biortogonal 4.4, também chamada de Wavelet 7-9)

#### Por que bior4.4?

A wavelet biortogonal 4.4 possui **fase linear** — os filtros de análise e síntese são distintos mas garantem reconstrução perfeita, sem distorção de fase. Amplamente utilizada no **padrão JPEG 2000**.

```python
import pywt

wavelet = pywt.Wavelet('bior4.4')
coeffs  = pywt.wavedec(signal, wavelet, level=5)
cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
```

#### Decomposição em 5 estágios

```
Nível  Subbanda         Conteúdo principal
─────────────────────────────────────────────────────
cD1    [fs/4 , fs/2]    Transições abruptas e ruído
cD2    [fs/8 , fs/4]    Componentes de alta frequência
cD3    [fs/16, fs/8]    Componentes intermediárias
cD4    [fs/32, fs/16]   Componentes de baixa frequência
cD5    [fs/64, fs/32]   Variações lentas
cA5    [0    , fs/64]   Tendência de longo prazo
```

> **Resultado:** enquanto a DFT deste sinal produz 3 picos (50, 120, 300 Hz) **sem informação temporal**, a DWT localiza *quando* cada componente está ativa — informação fundamental para análise de sinais não estacionários.

---

### 🔇 Questão 4 — Remoção de Ruído com Wavelets

**Notebook:** [`Questão4.ipynb`](Questão4.ipynb)

**Sinal:** `leleccum.mat` · **Wavelet:** `db4` (Daubechies 4)

#### Pipeline de denoising

```
Sinal ruidoso
      │
      ▼
 DWT (db4, 5 estágios)
      │
      ▼
 Limiarização dos coeficientes
  ├─ Hard:  |coef| < λ → 0  (preserva bordas, pode oscilar)
  └─ Soft:  |coef| < λ → 0, senão shrink por λ  (mais suave)
      │
      ▼
 Reconstrução (IDWT)
      │
      ▼
 Sinal denoised
```

#### Limiar de Donoho-Johnstone (Universal)

$$\lambda = \hat{\sigma} \sqrt{2 \ln N}, \quad \hat{\sigma} = \frac{\text{median}(|cD_1|)}{0.6745}$$

#### Denoising Wavelet vs. Filtragem pela DFT

| Critério | 🌊 Wavelet (db4) | 📊 DFT (Passa-Baixas) |
|----------|:---:|:---:|
| Localização temporal | ✅ Preserva transientes | ❌ Efeito de Gibbs |
| Remoção de ruído branco | ✅ Adaptativa por subbanda | ✅ Eficiente |
| Preservação de bordas | ✅ Coeficientes de detalhe | ❌ Truncamento espectral |
| Sinais não estacionários | ✅ Excelente | ⚠️ Limitada |
| Complexidade | `O(N)` | `O(N log N)` |

> **Conclusão:** para o sinal `leleccum`, a limiarização wavelet preserva melhor as bordas e transições do sinal útil, sem introduzir o *ringing* característico da filtragem ideal por DFT.

---

## 📁 Estrutura do Repositório

```
Pratica-5-PROCSIN/
│
├── 📓 Questao_1.ipynb          # Hadamard, CDMA, BER, error floor
├── 📓 Questão2.ipynb           # Resumo teórico sobre Wavelets
├── 📓 Questão3.ipynb           # Decomposição bior4.4, sinal não estacionário
├── 📓 Questão4.ipynb           # Denoising db4, comparação com DFT
│
└── 📄 Aula_Prática_5.pdf       # Enunciado oficial da prática
```

---

## ▶️ Como Executar

### Opção 1 — Google Colab (recomendado)

Clique no badge abaixo para abrir diretamente no Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeanraposojesus-hue/Pratica-5-PROCSIN/blob/main/Questao_1.ipynb)

### Opção 2 — Execução local

```bash
# 1. Clone o repositório
git clone https://github.com/jeanraposojesus-hue/Pratica-5-PROCSIN.git
cd Pratica-5-PROCSIN

# 2. Instale as dependências
pip install numpy scipy matplotlib pywavelets jupyter

# 3. Inicie o Jupyter
jupyter notebook
```

---

## 📦 Dependências

| Biblioteca | Versão mín. | Uso |
|------------|:-----------:|-----|
| `numpy` | 1.23+ | Operações matriciais e FFT |
| `scipy` | 1.9+ | Hadamard, carregamento de .mat |
| `matplotlib` | 3.6+ | Visualização de sinais e espectros |
| `pywavelets` | 1.4+ | DWT, famílias de wavelets, denoising |
| `jupyter` | 6.0+ | Ambiente de notebooks |

---

## 👥 Equipe

<table>
  <tr>
    <td align="center">
      <b>Jean Raposo Soares de Jesus</b><br/>
      <a href="https://github.com/jeanraposojesus-hue">@jeanraposojesus-hue</a>
    </td>
    <td align="center">
      <b>Igor Pinheiro Torres</b>
    </td>
    <td align="center">
      <b>Fabio Luiz Cocco Favre da Silva</b>
    </td>
  </tr>
</table>

---

<div align="center">

**Disciplina:** Processamento de Sinais I · **Professor:** Rafael Chaves  
**Curso:** Engenharia de Controle e Automação · **Instituição:** CEFET/RJ  

📚 Repositório de referência do professor: [rafaelschaves/gele7317-proc-sin](https://github.com/rafaelschaves/gele7317-proc-sin)

</div>
