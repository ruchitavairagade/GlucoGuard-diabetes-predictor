from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import io
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
    with open("static/index.html") as file:
        return file.read()

# Endpoint to display a plot dynamically
@app.post("/display-plot/")
async def display_plot():
    # Create a sample plot (replace this with your actual plot code)
    plt.figure()
    plt.plot([0, 1], [0, 1])  # Example plot
    plt.title("Dynamic Plot")

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    # Return the image as a response
    return StreamingResponse(buf, media_type="image/png")
