# Changelog

All notable changes to this project will be documented in this file.

## [2.1.0] - 2025-12-30

### Added
- **08-research-innovation agent**: Production-grade research methodology agent (665 lines)
  - Research methodology and experimental design
  - Literature review and paper analysis
  - Scientific paper writing and publication workflows
  - Reproducible research practices
  - Benchmark creation and evaluation
  - Ablation studies framework
- **reinforcement-learning skill**: Complete RL implementation guide (603 lines)
  - Q-Learning and DQN implementations
  - Policy Gradient methods (REINFORCE, PPO, A3C)
  - Multi-Agent Reinforcement Learning
  - Reward shaping and curriculum learning
  - Stable Baselines3 integration
- **time-series skill**: Comprehensive time series analysis (574 lines)
  - ARIMA/SARIMA modeling
  - Prophet forecasting
  - LSTM and Transformer architectures
  - Anomaly detection algorithms
  - Feature engineering for temporal data

### Changed
- All 8 agents now include dedicated **Troubleshooting** sections
  - Debug checklists for common issues
  - Domain-specific solutions and recovery strategies
- Updated plugin.json to register all 8 agents and 12 skills
- Enhanced marketplace.json with complete feature descriptions

### Fixed
- Missing 08-research-innovation agent registration in plugin.json
- Incomplete reinforcement-learning skill (was 23 lines, now 603)
- Incomplete time-series skill (was 23 lines, now 574)

### Quality Assurance
- Integrity Check: 8/8 agents, 12/12 skills verified
- SASMP v1.3.0 compliance confirmed
- EQHM (Event Quality and Health Monitoring) enabled
- Zero broken links, orphan skills, or circular dependencies

## [2.0.0] - 2025-12-28

### Added
- SASMP v1.3.0 compliance
- Template v2.1.0 README format
- Proper marketplace.json configuration
- Golden Format skill structure validation
- EQHM (Event Quality and Health Monitoring) enabled

### Changed
- Updated plugin.json to new format (paths instead of objects)
- Repository field now string format (E307 fix)
- Author field now object format
- Standardized hooks.json format
- Updated installation commands to `/plugin` format
- Restructured documentation to 18-section template

### Fixed
- E307: Repository field format
- E303: Marketplace name collision prevention
- Invalid hooks configuration format

## [1.0.0] - 2024-11-18

### Added
- Initial release
- 7 specialized agents (Programming, Math/Stats, Data Engineering, ML/AI, Visualization, MLOps, Career)
- 10 comprehensive skill modules
- 5 interactive commands (/learn, /browse-agent, /assess, /roadmap, /projects)
- 1200+ hours of curated content
- 50+ hands-on projects
- 12-month structured curriculum
- Based on roadmap.sh AI Data Scientist roadmap

---

**Format:** Based on [Keep a Changelog](https://keepachangelog.com/)
**Versioning:** [Semantic Versioning](https://semver.org/)
