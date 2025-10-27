from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
from inference import analyze_image

app = FastAPI(title="Tree Analyzer API")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    p1x: Optional[int] = Form(None),
    p1y: Optional[int] = Form(None),
    p2x: Optional[int] = Form(None),
    p2y: Optional[int] = Form(None),
):
    data = await image.read()
    points = None
    if all(v is not None for v in [p1x, p1y, p2x, p2y]):
        points = ((p1x, p1y), (p2x, p2y))
    result = analyze_image(data, manual_points=points)
    return result
