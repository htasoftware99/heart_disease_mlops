# Heart Disease Prediction MLOps Project

A comprehensive, production ready MLOps system for predicting heart disease using clinical patient data. This project integrates the entire machine learning lifecycle from data ingestion and versioning to cloud deployment and automated CI/CD pipelines.

## 🏗️ System Architecture

<p align="center">
  <img src="structure.png" width="700"/>
</p>

## 🌟 Key Features

- **End-to-End Pipeline:** Automated data ingestion, feature engineering, and model training via DVC.
- **Experiment Tracking:** Comprehensive logging of metrics and parameters using MLflow.
- **Data & Model Versioning:** Artifact management and lineage tracking with DVC and Google Cloud Storage.
- **Robust CI/CD:** Multi-stage Jenkins pipeline including static analysis (SonarQube) and security scanning (Trivy).
- **GitOps Deployment:** Continuous delivery to Kubernetes orchestrated by Argo CD.
- **Infrastructure as Code (IaC):** Cloud resource provisioning using Terraform.
- **Interactive Web UI:** User friendly Flask application for real time heart disease risk assessment.

## 🛠 Technology Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **ML Frameworks** | Scikit-learn, XGBoost, Pandas, NumPy |
| **Orchestration** | DVC (Data Version Control) |
| **Tracking** | MLflow |
| **Infrastructure** | Terraform, Google Cloud Platform (GCP), Google Cloud Stroge (Bucket), Google Compute Emgine (VM) |
| **CI/CD & GitOps** | Jenkins, Argo CD |
| **Security & Quality**| SonarQube, Trivy |
| **Containerization** | Docker, Kubernetes (K8s), Minikube, Kubectl |
| **Web Service** | Flask, HTML5, CSS3 |

## 📁 Project Structure

```text
├── .dvc/                   # DVC configuration and metadata
├── artifacts/              # Pipeline outputs (Raw/Processed data, Models)
├── config/                 # Project & path configurations (YAML, Python)
├── infrastructure/         # Terraform HCL scripts for GCP provisioning
├── manifests/              # Kubernetes deployment and service YAMLs
├── src/                    # Core Python modules
│   ├── data_ingestion.py   # GCP bucket data retrieval & splitting
│   ├── data_processing.py  # Cleaning, scaling, and OHE transformation
│   ├── model_training.py   # Training, evaluation & MLflow logging
│   ├── logger.py           # Structured logging
│   └── custom_exception.py # Centralized error handling
├── static/ & templates/    # Flask UI assets
├── app.py                  # Web application entry point
├── dvc.yaml                # DVC pipeline stage definitions
├── Jenkinsfile             # Jenkins CI/CD pipeline script
└── requirements.txt        # Python dependency list
```

## ⚙️ Installation & Setup

1. **Clone & Environment Setup:**
   ```bash
   git clone <repo-url>
   cd heart_disease_mlops
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **DVC Initialization:**
   ```bash
   dvc init --no-scm  # Pull artifacts if remote is configured
   dvc repro # Run the pipeline to generate artifacts locally
   ```

3. **GCP Configuration:**
   Ensure you have your GCP credentials configured for Terraform and DVC GCS remote access.

## 🚀 Usage

### 1. Training Pipeline
Execute the full DVC pipeline:
```bash
dvc repro
```
This triggers:
1. `data_ingestion`: Downloads data from GCS bucket.
2. `data_processing`: Handles missing values, scaling, and encoding.
3. `model_training`: Trains the model and logs to MLflow.

### 2. Local Web Server
Start the Flask application:
```bash
python app.py
```
Visit `http://localhost:5000` to interact with the prediction interface.

## 🔄 CI/CD & GitOps Workflow

The project follows a modern DevOps workflow:

- **SonarQube Analysis:** Automated code quality checks triggered on every commit.
- **Dockerization:** Automated builds of the application image.
- **Trivy Security Scan:** Scans the Docker image for OS-level and dependency vulnerabilities.
- **Docker Hub:** Pushes the verified image to the registry.
- **Argo CD Sync:** Automatically detects changes in the Kubernetes manifests and synchronizes the cluster state to match the Git repository.

## 🏗 Infrastructure Provisioning

The `infrastructure/` directory contains Terraform code to set up:
- **GCS Buckets:** For data and model storage.
- **Compute Instances:** For hosting the application or runners.
- **IAM Roles:** For secure access management.

To deploy infrastructure:
```bash
cd infrastructure
terraform init
terraform fmt
terraform plan
terraform apply
terraform destroy # do not forget to run final
```