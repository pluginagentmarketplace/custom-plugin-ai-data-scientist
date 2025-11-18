# AI & Data Scientist Plugin 🚀

**Ultra-comprehensive learning system** for mastering AI, Machine Learning, and Data Science from beginner to expert level. Based on the official [roadmap.sh AI Data Scientist roadmap](https://roadmap.sh/ai-data-scientist) with **1200+ hours** of curated content, **7 specialized agents**, **10+ skills**, and **50+ hands-on projects**.

## 🎯 Plugin Overview

This plugin transforms your Claude Code experience into a complete AI & Data Science learning environment with:

- **7 Specialized Agents**: Each an expert in a specific domain
- **10 Invokable Skills**: Instant access to practical knowledge
- **5 Slash Commands**: Interactive learning paths
- **50+ Projects**: Hands-on portfolio building
- **Complete Roadmap**: 12-month structured curriculum
- **Assessment System**: Track your progress

## 📦 Installation

### One-Command Install
```bash
# Load plugin in Claude Code
# Simply reference the plugin directory
./custom-plugin-ai-data-scientist
```

### From GitHub
```bash
# Clone repository
git clone https://github.com/pluginagentmarketplace/custom-plugin-ai-data-scientist.git

# Load in Claude Code
cd custom-plugin-ai-data-scientist
```

## 🤖 7 Specialized Agents

### 1. Programming Foundations Expert
**Focus:** Python, R, SQL, Git, Data Structures
- Master Python for data science
- SQL query optimization
- Version control best practices
- Production-ready code

### 2. Mathematics & Statistics Specialist
**Focus:** Linear Algebra, Calculus, Probability, Statistics
- Statistical inference and hypothesis testing
- Mathematical foundations for ML
- A/B testing and experimental design
- Bayesian statistics

### 3. Data Engineering & Processing Expert
**Focus:** ETL/ELT, Big Data, Spark, Kafka
- Build scalable data pipelines
- Apache Spark for big data
- Data warehousing (Snowflake, BigQuery)
- Stream processing

### 4. Machine Learning & AI Specialist
**Focus:** ML Algorithms, Deep Learning, NLP, Computer Vision
- Supervised/unsupervised learning
- Deep learning (CNNs, RNNs, Transformers)
- NLP and Computer Vision
- Model optimization

### 5. Data Visualization & Communication Expert
**Focus:** EDA, Dashboards, Storytelling, BI Tools
- Exploratory data analysis
- Interactive dashboards (Plotly, Dash)
- Tableau, Power BI
- Stakeholder communication

### 6. MLOps & Deployment Specialist
**Focus:** Docker, Kubernetes, CI/CD, Cloud Platforms
- Model deployment strategies
- Containerization and orchestration
- CI/CD pipelines for ML
- Production monitoring

### 7. Domain Knowledge & Career Advisor
**Focus:** Business Acumen, Ethics, Career Development
- Industry applications (finance, healthcare, retail)
- Ethics and responsible AI
- Interview preparation
- Portfolio building

## 🎓 10 Invokable Skills

Access instant knowledge on demand:

1. **`python-programming`** - Python fundamentals to advanced
2. **`statistical-analysis`** - Hypothesis testing, A/B testing
3. **`data-engineering`** - ETL pipelines, Spark, data quality
4. **`machine-learning`** - Scikit-learn, model selection
5. **`deep-learning`** - PyTorch, TensorFlow, neural networks
6. **`nlp-processing`** - Text analysis, LLMs, Transformers
7. **`computer-vision`** - CNNs, object detection, segmentation
8. **`data-visualization`** - Matplotlib, Seaborn, Plotly, BI tools
9. **`mlops-deployment`** - Docker, Kubernetes, model serving
10. **`model-optimization`** - Quantization, pruning, AutoML

## 🔧 5 Slash Commands

### `/learn`
Start your personalized learning journey
- Choose path: Beginner, Intermediate, Advanced
- 12-month roadmap with weekly goals
- Study schedule templates
- Learning resources

### `/browse-agent`
Explore all 7 specialized agents
- Detailed agent capabilities
- When to use each agent
- Learning progression paths
- Specialization tracks

### `/assess`
Evaluate your knowledge across all domains
- Self-assessment questionnaire
- Score interpretation (0-140 points)
- Skill gap analysis
- Personalized learning plan

### `/roadmap`
View complete AI & Data Scientist roadmap
- 12-month curriculum
- Phase-by-phase breakdown
- Resources by category
- Success metrics

### `/projects`
Browse 50+ hands-on projects
- Beginner to advanced levels
- Domain-specific projects
- Portfolio building guide
- Step-by-step implementations

## 🚀 Quick Start

### For Complete Beginners
```bash
# 1. Start with learning path
/learn

# 2. Choose "Complete Beginner" track

# 3. Begin with Programming Foundations Agent
"I need help learning Python from scratch"

# 4. Practice with beginner projects
/projects  # Select Titanic or House Prices
```

### For Intermediate Learners
```bash
# 1. Assess your current skills
/assess

# 2. View roadmap
/roadmap

# 3. Focus on weak areas
"Help me with deep learning using PyTorch"

# 4. Build advanced projects
/projects  # Image classification, NLP
```

### For Career Transitioners
```bash
# 1. Assess skills
/assess

# 2. Work with Domain & Career Agent
"Help me build a data science portfolio"

# 3. Interview preparation
"Prepare me for ML engineering interviews"

# 4. Projects for resume
/projects  # End-to-end MLOps, production systems
```

## 📚 Example Workflows

### Workflow 1: Build Your First ML Model
```
1. /learn → Choose "Complete Beginner"
2. Use Programming Foundations Agent for Python basics
3. Use Machine Learning Agent for first model
4. /projects → Titanic Survival Prediction
5. Deploy with MLOps Agent
```

### Workflow 2: Specialize in NLP
```
1. /assess → Evaluate current skills
2. Use Deep Learning Agent → NLP focus
3. Invoke `nlp-processing` skill for quick reference
4. /projects → Sentiment Analysis, Text Classification
5. Advanced: Chatbot with Transformers
```

### Workflow 3: Become MLOps Engineer
```
1. /roadmap → View MLOps path
2. Use MLOps & Deployment Agent
3. Invoke `mlops-deployment` skill
4. /projects → End-to-End MLOps Pipeline
5. Deploy to cloud (AWS/GCP/Azure)
```

## 🎯 Learning Paths

### Path 1: Data Scientist (12 months)
```
Months 1-3: Foundations (Python, SQL, Statistics)
Months 4-6: Machine Learning (Scikit-learn, projects)
Months 7-9: Deep Learning (PyTorch, specialization)
Months 10-12: Production & Career (MLOps, portfolio)
```

### Path 2: ML Engineer (12 months)
```
Months 1-3: Programming + Data Engineering
Months 4-6: ML + Model Optimization
Months 7-9: Deep Learning + Advanced ML
Months 10-12: MLOps + Production Systems
```

### Path 3: NLP Specialist (After core skills)
```
Foundations → ML Basics → Deep Learning →
NLP Fundamentals → Transformers & LLMs →
Fine-tuning → Production NLP Systems
```

## 📊 Plugin Structure

```
custom-plugin-ai-data-scientist/
├── .claude-plugin/
│   └── plugin.json              # Plugin manifest
├── agents/                      # 7 specialized agents
│   ├── 01-programming-foundations.md
│   ├── 02-mathematics-statistics.md
│   ├── 03-data-engineering.md
│   ├── 04-machine-learning-ai.md
│   ├── 05-visualization-communication.md
│   ├── 06-mlops-deployment.md
│   └── 07-domain-career.md
├── skills/                      # 10 invokable skills
│   ├── python-programming/SKILL.md
│   ├── statistical-analysis/SKILL.md
│   ├── data-engineering/SKILL.md
│   ├── machine-learning/SKILL.md
│   ├── deep-learning/SKILL.md
│   ├── nlp-processing/SKILL.md
│   ├── computer-vision/SKILL.md
│   ├── data-visualization/SKILL.md
│   ├── mlops-deployment/SKILL.md
│   └── model-optimization/SKILL.md
├── commands/                    # 5 slash commands
│   ├── learn.md
│   ├── browse-agent.md
│   ├── assess.md
│   ├── roadmap.md
│   └── projects.md
├── hooks/
│   └── hooks.json               # Automation hooks
└── README.md
```

## 🔥 Features

✅ **Comprehensive Coverage**: 1200+ hours of content
✅ **Practical Focus**: 50+ hands-on projects
✅ **Structured Learning**: 12-month roadmap
✅ **Self-Paced**: Learn at your own speed
✅ **Assessment System**: Track progress
✅ **Production-Ready**: Deploy real ML systems
✅ **Career Guidance**: Interview prep, portfolio
✅ **Modern Stack**: Latest tools and frameworks
✅ **Best Practices**: Industry standards
✅ **Community-Driven**: Based on roadmap.sh

## 🛠️ Tech Stack Covered

**Programming:**
- Python, R, SQL
- Git/GitHub
- Data structures & algorithms

**Data Science:**
- Pandas, NumPy, SciPy
- Matplotlib, Seaborn, Plotly
- Jupyter Notebooks

**Machine Learning:**
- Scikit-learn
- XGBoost, LightGBM
- Auto-sklearn, H2O

**Deep Learning:**
- PyTorch, TensorFlow
- Hugging Face Transformers
- YOLO, U-Net

**Big Data:**
- Apache Spark (PySpark)
- Apache Kafka
- Hadoop ecosystem

**MLOps:**
- Docker, Kubernetes
- FastAPI, Flask
- MLflow, DVC
- Prometheus, Grafana

**Cloud Platforms:**
- AWS (SageMaker, EC2, S3)
- Google Cloud (Vertex AI, BigQuery)
- Azure (Azure ML, Synapse)

**BI Tools:**
- Tableau
- Power BI
- Looker

## 📈 Success Metrics

After completing this plugin's curriculum:

**Technical Skills:**
- ✅ Build end-to-end ML pipelines
- ✅ Deploy production ML systems
- ✅ Process big data with Spark
- ✅ Deep learning with PyTorch/TensorFlow
- ✅ NLP and Computer Vision projects

**Portfolio:**
- ✅ 5+ polished GitHub projects
- ✅ Deployed ML applications
- ✅ Kaggle competition participation
- ✅ Technical blog posts

**Career:**
- ✅ Data Scientist/ML Engineer ready
- ✅ Interview-ready (coding, ML, case studies)
- ✅ Professional network
- ✅ Industry knowledge

## 🤝 Contributing

This plugin is based on the community-driven [roadmap.sh AI Data Scientist roadmap](https://roadmap.sh/ai-data-scientist). Contributions are welcome!

## 📝 License

MIT License

## 🌟 Acknowledgments

- Based on [roadmap.sh AI Data Scientist Roadmap](https://roadmap.sh/ai-data-scientist)
- Inspired by the data science and ML community
- Built for Claude Code users

## 🎓 Start Learning Today!

```bash
# Start your journey
/learn

# Or assess your skills
/assess

# Or dive into a project
/projects
```

---

**Remember:** Becoming an AI & Data Scientist is a journey, not a destination. This plugin is your guide, but your dedication and practice are what will make you successful. Start small, build consistently, and never stop learning!

**Ready to transform your career? Let's begin! 🚀**
