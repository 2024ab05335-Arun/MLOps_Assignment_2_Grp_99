# MLOps_Assignment_2_Grp_99

python .\preprocess_petimages.py

# Run using the locally saved model (baseline_cnn_best.h5 must exist)
python inference_service.py --port 8000

# Or load model from MLflow run
python inference_service.py --port 8000 --model-uri "runs:/<run_id>/model"

# Optionally supply labels JSON (list of class names)
python inference_service.py --port 8000 --model-uri "runs:/<run_id>/model" --labels labels.json

python inference_service.py --port 8000 --model-uri "runs:/<run_id>/model"

curl -X POST "http://localhost:8000/predict" -F "files=@/path/to/image.jpg"

B64=$(base64 -w 0 image.jpg) && curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"instances\":[\"$B64\"]}"

python inference_client.py --image /path/to/image.jpg --mode file --url http://localhost:8000
# or
python inference_client.py --image /path/to/image.jpg --mode base64 --url http://localhost:8000



cd MLOps_Assignment_2_Grp_99 ; C:/ArunDocs/Code/.venv/Scripts/python.exe -m pytest -q tests -q

cd ..\MLOps_Assignment_2_Grp_99 ; C:/ArunDocs/Code/.venv/Scripts/python.exe -m pytest tests -vv > pytest_output.txt 2>&1


Docker:

Build:
cd c:/ArunDocs/Code/MLOps_Assignment_2_Grp_99
docker build -t pet-inference:latest .

Run (model is copied into the image from baseline_cnn_best.h5):
docker run -p 8000:8000 --rm pet-inference:latest

Verify with curl (health):
curl http://localhost:8000/health

Verify with curl (multipart file upload):
curl -X POST "http://localhost:8000/predict" -F "files=@data/processed/PetImages_224_split/test/Cat/<your-image>.jpg"

python inference_client.py --image data/processed/PetImages_224_split/test/Cat/<your-image>.jpg --mode file --url http://localhost:8000

# build image
cd c:/ArunDocs/Code/MLOps_Assignment_2_Grp_99
docker build -t pet-inference:latest .

# run using an MLflow model URI (recommended)
docker run -p 8000:8000 --rm -e MODEL_URI="runs:/<run_id>/model" pet-inference:latest

# or run using the bundled local model if present
docker run -p 8000:8000 --rm pet-inference:latest

Verify endpoints:
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" -F "files=@data/processed/PetImages_224_split/test/Cat/<image>.jpg"
