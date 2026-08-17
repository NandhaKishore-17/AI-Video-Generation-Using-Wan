import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

def main():
    load_dotenv()
    
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN is not configured in .env")
        return
        
    print(f"HF TOKEN LOADED: {bool(token)}")
    print(f"HF TOKEN LENGTH: {len(token)}")
    
    provider = os.getenv("HF_PROVIDER", "fal-ai")
    model = os.getenv("HF_VIDEO_MODEL", "Wan-AI/Wan2.2-TI2V-5B")
    
    print(f"PROVIDER: {provider}")
    print(f"MODEL: {model}")
    print("Starting generation...")
    
    client = InferenceClient(
        provider=provider,
        api_key=token
    )
    
    try:
        video_bytes = client.text_to_video(
            "A cinematic ancient laboratory filled with glowing blue runes and mysterious energy, slow camera movement",
            model=model,
            num_frames=24
        )
        
        output_file = "test_wan_video.mp4"
        with open(output_file, "wb") as f:
            f.write(video_bytes)
            
        print("SUCCESS")
        print(f"Saved: {output_file}")
        print(f"Size: {os.path.getsize(output_file)} bytes")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    main()
