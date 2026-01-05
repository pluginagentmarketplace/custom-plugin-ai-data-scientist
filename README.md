<div align="center">

<!-- Animated Typing Banner -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=2E9EF7&center=true&vCenter=true&multiline=true&repeat=true&width=600&height=100&lines=Ai+Data+Scientist+Assistant;8+Agents+%7C+12+Skills;Claude+Code+Plugin" alt="Ai Data Scientist Assistant" />

<br/>

<!-- Badge Row 1: Status Badges -->
[![Version](https://img.shields.io/badge/Version-2.1.0-blue?style=for-the-badge)](https://github.com/pluginagentmarketplace/custom-plugin-ai-data-scientist/releases)
[![License](https://img.shields.io/badge/License-Custom-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen?style=for-the-badge)](#)
[![SASMP](https://img.shields.io/badge/SASMP-v1.3.0-blueviolet?style=for-the-badge)](#)

<!-- Badge Row 2: Content Badges -->
[![Agents](https://img.shields.io/badge/Agents-8-orange?style=flat-square&logo=robot)](#-agents)
[![Skills](https://img.shields.io/badge/Skills-12-purple?style=flat-square&logo=lightning)](#-skills)
[![Commands](https://img.shields.io/badge/Commands-5-green?style=flat-square&logo=terminal)](#-commands)

<br/>

<!-- Quick CTA Row -->
[📦 **Install Now**](#-quick-start) · [🤖 **Explore Agents**](#-agents) · [📖 **Documentation**](#-documentation) · [⭐ **Star this repo**](https://github.com/pluginagentmarketplace/custom-plugin-ai-data-scientist)

---

### What is this?

> **Ai Data Scientist Assistant** is a production-grade Claude Code plugin with **8 agents** and **12 skills** for AI & Data Science development. SASMP v1.3.0 compliant with EQHM enabled.

</div>

---

## 📑 Table of Contents

<details>
<summary>Click to expand</summary>

- [Quick Start](#-quick-start)
- [Features](#-features)
- [Agents](#-agents)
- [Skills](#-skills)
- [Commands](#-commands)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

</details>

---

## 🚀 Quick Start

### Prerequisites

- Claude Code CLI v2.0.27+
- Active Claude subscription

### Installation (Choose One)

<details open>
<summary><strong>Option 1: From Marketplace (Recommended)</strong></summary>

```bash
# Step 1️⃣ Add the marketplace
/plugin marketplace add pluginagentmarketplace/custom-plugin-ai-data-scientist

# Step 2️⃣ Install the plugin
/plugin install ai-data-scientist-plugin@pluginagentmarketplace-ai-data-scientist

# Step 3️⃣ Restart Claude Code
# Close and reopen your terminal/IDE
```

</details>

<details>
<summary><strong>Option 2: Local Installation</strong></summary>

```bash
# Clone the repository
git clone https://github.com/pluginagentmarketplace/custom-plugin-ai-data-scientist.git
cd custom-plugin-ai-data-scientist

# Load locally
/plugin load .

# Restart Claude Code
```

</details>

### ✅ Verify Installation

After restart, you should see these agents:

```
ai-data-scientist-plugin:02-mathematics-statistics
ai-data-scientist-plugin:07-domain-career
ai-data-scientist-plugin:03-data-engineering
ai-data-scientist-plugin:04-machine-learning-ai
ai-data-scientist-plugin:05-visualization-communication
... and 2 more
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **8 Agents** | Specialized AI agents with troubleshooting sections |
| 🛠️ **12 Skills** | Production-grade capabilities including RL & Time Series |
| ⌨️ **5 Commands** | Quick slash commands for learning & assessment |
| 🔄 **SASMP v1.3.0** | Full protocol compliance with EQHM enabled |
| 🔧 **Troubleshooting** | Built-in debug checklists & solutions for each agent |

---

## 🤖 Agents

### 8 Specialized Agents

| # | Agent | Purpose |
|---|-------|---------|
| 1 | **01-python-data-science** | Python, R, SQL, Git, data structures, algorithms |
| 2 | **02-mathematics-statistics** | Linear algebra, calculus, probability, statistics |
| 3 | **03-data-engineering** | ETL/ELT pipelines, Spark, Hadoop, data lakes |
| 4 | **04-machine-learning-ai** | ML/DL, NLP, computer vision, model optimization |
| 5 | **05-visualization-communication** | EDA, dashboards, storytelling, BI tools |
| 6 | **06-mlops-deployment** | Docker, Kubernetes, CI/CD, monitoring |
| 7 | **07-domain-career** | Business acumen, ethics, career development |
| 8 | **08-research-innovation** | Research methodology, paper writing, experiments |

---

## 🛠️ Skills

### Available Skills

| Skill | Description |
|-------|-------------|
| `python-programming` | Python fundamentals, data structures, OOP, Pandas, NumPy |
| `statistical-analysis` | Probability, distributions, hypothesis testing, A/B testing |
| `data-engineering` | ETL pipelines, Apache Spark, data warehousing, streaming |
| `machine-learning` | Supervised/unsupervised learning, scikit-learn, evaluation |
| `deep-learning` | Neural networks, CNNs, RNNs, Transformers, PyTorch |
| `nlp-processing` | Text processing, sentiment analysis, LLMs, BERT |
| `computer-vision` | Image classification, object detection, segmentation |
| `data-visualization` | EDA, Matplotlib, Seaborn, Plotly, dashboards |
| `mlops-deployment` | Docker, Kubernetes, CI/CD, model monitoring |
| `model-optimization` | Quantization, pruning, AutoML, hyperparameter tuning |
| `reinforcement-learning` | Q-learning, DQN, PPO, multi-agent systems, Gym |
| `time-series` | ARIMA, Prophet, forecasting, anomaly detection |

---

## ⌨️ Commands

| Command | Description |
|---------|-------------|
| `/learn` | Start Your AI & Data Scientist Learning Journey |
| `/assess` | AI & Data Scientist Knowledge Assessment |
| `/browse-agent` | Explore AI & Data Scientist Agents |
| `/projects` | 50+ Hands-On AI & Data Science Projects |
| `/roadmap` | AI & Data Scientist Complete Roadmap 2025 |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [LICENSE](LICENSE) | License information |

---

## 📁 Project Structure

<details>
<summary>Click to expand</summary>

```
custom-plugin-ai-data-scientist/
├── 📁 .claude-plugin/
│   ├── plugin.json
│   └── marketplace.json
├── 📁 agents/              # 8 agents with troubleshooting
├── 📁 skills/              # 12 skills (Production-grade)
├── 📁 commands/            # 5 commands
├── 📁 hooks/
├── 📄 README.md
├── 📄 CHANGELOG.md
└── 📄 LICENSE
```

</details>

---

## 📅 Metadata

| Field | Value |
|-------|-------|
| **Version** | 2.1.0 |
| **Last Updated** | 2025-12-30 |
| **Status** | Production Ready |
| **SASMP** | v1.3.0 |
| **EQHM** | Enabled |
| **Agents** | 8 |
| **Skills** | 12 |
| **Commands** | 5 |

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

1. Fork the repository
2. Create your feature branch
3. Follow the Golden Format for new skills
4. Submit a pull request

---

## ⚠️ Security

> **Important:** This repository contains third-party code and dependencies.
>
> - ✅ Always review code before using in production
> - ✅ Check dependencies for known vulnerabilities
> - ✅ Follow security best practices
> - ✅ Report security issues privately via [Issues](../../issues)

---

## 📝 License

Copyright © 2025 **Dr. Umit Kacar** & **Muhsin Elcicek**

Custom License - See [LICENSE](LICENSE) for details.

---

## 👥 Contributors

<table>
<tr>
<td align="center">
<strong>Dr. Umit Kacar</strong><br/>
Senior AI Researcher & Engineer
</td>
<td align="center">
<strong>Muhsin Elcicek</strong><br/>
Senior Software Architect
</td>
</tr>
</table>

---

<div align="center">

**Made with ❤️ for the Claude Code Community**

[![GitHub](https://img.shields.io/badge/GitHub-pluginagentmarketplace-black?style=for-the-badge&logo=github)](https://github.com/pluginagentmarketplace)

</div>
