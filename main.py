from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = "aich007/T5-small-title-generation"
model = None
tokenizer = None


async def load_model():
    """Load model without quantization for Render compatibility."""
    global model, tokenizer
    try:
        print("Loading model...")

        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        model.eval()

        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on application startup."""
    print("Starting up...")

    success = await load_model()
    if not success:
        print("WARNING: Model failed to load!")
    yield
    print("Application shutdown!")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_title(message, max_new_tokens=10):
    """Generate title from message using the loaded model."""
    try:
        inputs = tokenizer(
            message, return_tensors="pt", truncation=True, max_length=128
        )

        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
        )

        generated_title = tokenizer.decode(
            output_sequences[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return generated_title
    except Exception as e:
        print(f"Error generating title: {e}")
        return f"Error: {str(e)}"


@app.get("/get_title")
async def get_title(message: str = "write a c++ code to add two numbers"):
    """Generate title for a given message."""
    if model is None or tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded yet. Please try again."},
        )

    try:
        title = generate_title(message)
        return {"message": message, "title": title}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/test")
async def test():
    """Test endpoint to verify model functionality."""
    if model is None or tokenizer is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    print("Testing model...")
    my_message = "write a c++ code to add two numbers"

    t0 = time.time()
    generated_title = generate_title(my_message)
    inference_time = time.time() - t0

    print(f"Inference time: {inference_time:.3f}s")
    print(f"Original Message: {my_message}")
    print(f"Generated Title: {generated_title}")

    return {
        "message": my_message,
        "title": generated_title,
        "inference_time_seconds": round(inference_time, 3),
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if model is not None else "initializing",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
    }


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "T5 Title Generator API",
        "status": "online" if model is not None else "loading",
    }
