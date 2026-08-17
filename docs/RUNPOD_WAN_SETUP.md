# Runpod Serverless Wan2.2 Video Generation Setup

This application has been upgraded to offload heavy Wan2.2-TI2V-5B video generation entirely to Runpod Serverless. This ensures that your local Windows machine remains lightweight and never needs to download or run demanding local inference workloads (e.g., PyTorch, CUDA, Diffusers).

Follow these steps to configure your Runpod environment and connect it to the backend.

## 1. Create a Runpod Account
If you don't already have one, create an account at [Runpod.io](https://runpod.io).

## 2. Configure a Serverless Endpoint
1. Navigate to the **Serverless** tab in your Runpod dashboard.
2. Select an existing Wan2.2 Serverless Template or deploy a worker container capable of running the `Wan-AI/Wan2.2-TI2V-5B` model.
3. For the GPU configuration, ensure you select an instance with enough VRAM (at least 24GB+ is recommended for video generation).
4. **Cost Safety**: To avoid unnecessary charges, make sure the **Minimum Workers** setting is set to `0` (Scale to Zero) when you aren't using the app.

## 3. Obtain Your API Key and Endpoint URL
1. Navigate to **Settings > API Keys** in Runpod. Generate a new API key and copy it.
2. Go back to your Serverless Endpoint in Runpod. You will see an endpoint ID or a full URL provided by Runpod (e.g., `https://api.runpod.ai/v2/your-endpoint-id`). Copy this base URL.

## 4. Configure Your Backend
Open the `.env` file located in the `backend/` directory of this project.

Update the following variables using your copied values. **Do not wrap the URL in quotes.**

```env
RUNPOD_API_KEY=<your_key>
RUNPOD_WAN_ENDPOINT_URL=<your_endpoint>
WAN_MODEL=Wan2.2-TI2V-5B
```

*Note: Never commit your actual API key to version control or share it publicly.*

## 5. Start and Test the Backend
Start your FastAPI backend as usual:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You can test video generation from the frontend or by manually hitting the endpoint. The application will log:
```
STARTING REMOTE WAN2.2 VIDEO GENERATION VIA RUNPOD
Runpod generation submitted
Runpod job ID: <id>
...
Video saved successfully
```

## Important Note on Image-to-Video
If your scene prompt utilizes image-to-video capabilities, the image URL you provide to the application **must** be a publicly accessible `https://` URL (e.g., an S3 bucket URL). The remote Runpod GPU cannot access `localhost` or relative file paths from your laptop. If you attempt to send a local image, the backend will safely reject the request to prevent unexpected generation failures.
