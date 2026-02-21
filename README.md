#### MLOps_Assignment_2_Grp_99
- 1. Arun Gupta (2024AB05335)
- 2. Ankur Gupta (2024AB05337)
- 3. Arjun Chandran (2024AB05336)
- 4. Vineet Suresh Kumar Yadav (2024AB05340)
- 5. Umang Nathani (2024AB05334)

##### Data pre-processing and spiliting.
- Pre-process to 224x224 RGB images for standard CNNs. Split into train/validation/test sets (e.g., 80%/10%/10%).
    Files added:
    - `preprocess_petimages.py` — main preprocessing script
    - `data_split.py` — data split script
    Quick run:
        ```bash
        python preprocess_petimages.py \
        --input data/Dataset/PetImages \
        --output data/processed/PetImages_224 \
        --method resize_crop \
        --workers 8
        ```
        ```bash
        python data_split.py
        ```

##### M1:  Model Development & Experiment Tracking
- 1. Data & Code Versioning
    Use Git for source code versioning and Git‑LFS for dataset versioning and to track pre-processed data.
    File added to track data versioning:
    - `.gitattributes`
- 2. Model Building 
    Use simple CNN for model training and save the trained model in .keras format.
    File updated to track models(due to large model size):
    - `.gitattributes`
- 3. Experiment Tracking
    Use MLFlow to log runs, parameters, metrics, and artifacts (confusion matrix, loss curves) 
    Files added:
    - `preprmlflow_workflow.py` — Train model and configure MLFlow
        ```bash
        python mlflow_workflow.py
        python mlflow_workflow.py --epochs 10
        mlflow ui --backend-store-uri mlruns --port 5000
        ```

##### M2: Model Packaging & Containerization 
- 1. Inference Service
    Wrap the trained model with a simple REST API using FastAPI. 
    Files added:
    - `inference_service.py` — REST API to access model for inderence
    - `inference_client.py` — Inference client to validate the image
        ```bash
        python inference_service.py --port 8000 --model-uri "runs:/<run_id>/model"
        python inference_service.py --port 8000 --model-uri "runs:/a40b43613cdb458b93c2da1e0a723b02/model"
        python inference_client.py --image /path/to/image.jpg --mode file --url http://localhost:8000
        python inference_client.py --image /path/to/image.jpg --mode base64 --url http://localhost:8000
        ```
- 2. Environment Specification
    - `requirements.txt`
- 3. Containerization:
    - `Dockerfile`
    - `download_model.py`
    - `entrypoint.sh`
    - build image
        ```
        docker build -t pet-inference:latest .
        ```
    - run using an MLflow model URI (recommended)
        ```
        docker run -p 8000:8000 --rm -e MODEL_URI="runs:/<run_id>/model" pet-inference:latest
        ```
##### M3: CI Pipeline for Build, Test & Image Creation
- 1. Automated Testing
    Unit test using pytest
    Files added:
    - `tests/test_inference_utils.py` — REST API to access model for inderence
    - `tests/test_preprocess.py` — Inference client to validate the image
        ```bash
        python -m pytest -q tests -q
        python -m pytest tests -vv > pytest_output.txt 2>&1
        ```
- 2. CI Setup (Choose one: GitHub Actions / GitLab CI / Jenkins / Tekton) 
    Define a pipeline that on every push/merge request, checks out the repository, installs 
    dependencies, runs unit tests, and builds the Docker image 
- 3. Artifact Publishing:
    Configure the pipeline to push the Docker image to a container registry (e.g., Docker Hub, GitHub Container Registry, local registry).

##### M4: CD Pipeline & Deployment    
- 1. Deployment Target 
    Choose one: local Kubernetes cluster (kind/minikube/microk8s), Docker Compose, or a 
    simple VM server. 
    Define infrastructure manifests: For Kubernetes: Deployment + Service YAML. 
    For Docker Compose: docker-compose.yml. 
- 2. CD / GitOps Flow 
    Extend CI or use a CD tool (Argo CD, Jenkins, GitHub Actions environment) to: - - 
    Pull the new image from the registry. 
    Deploy/update the running service automatically on main branch changes. 
- 3. Smoke Tests / Health Check 
    Verify endpoints:
        ```
        curl http://localhost:8000/health
        curl -X POST "http://localhost:8000/predict" -F "files=@data/processed/PetImages_224_split/test/Cat/<image>.jpg"
        ```

##### M5: Monitoring, Logs & Final Submission –     
- 1. Basic Monitoring & Logging 
    Logging added to inference service and logs are created in:
    - `inference.log`
- 2. Model Performance Tracking (Post‑Deployment) 
    Collect a small batch of real or simulated requests and true labels.    