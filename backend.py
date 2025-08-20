from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import matplotlib.pyplot as plt
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Allow cross-origin requests (for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend's URL for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the frontend files (index.html, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html file
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join("static", "index.html")) as file:
        return file.read()

# Run Python code dynamically
@app.post("/run-code/")
async def run_code():
    def run():
        try:
            # Run the provided script (e.g., "script.py")
            process = subprocess.Popen(
                ["python", "main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            for line in process.stdout:
                yield f"{line.decode()}<br>"
            for line in process.stderr:
                yield f"ERROR: {line.decode()}<br>"
        except Exception as e:
            yield f"ERROR: {str(e)}<br>"

    return StreamingResponse(run(), media_type="text/html")

# Save and serve plots
@app.post("/save-plot/")
async def save_plot():
    # Create a sample plot (replace this with your actual plot code)
    plt.figure()
    plt.plot([0, 1], [0, 1])  # Example plot
    plot_path = "plot.png"
    plt.savefig(plot_path)
    plt.close()

    # Serve the saved plot to the frontend
    return FileResponse(plot_path)
