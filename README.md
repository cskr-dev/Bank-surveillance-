# Video Annotation Server

This project provides tools for video annotation using trained YOLO models and a Django-based web interface.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python manage.py migrate
python manage.py runserver
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser, upload a video, and download the annotated video.

## Annotation Pipelines

- **Infer + SORT**: bounding-box and tracking-based annotation (`video_infer_sort.py`).
- **Human-Tool Safety**: safety alerts for tool usage (`video_human_tool_safety.py`).
- **Person Detection**: human/person detection only (frames annotated per person bounding box).

Uploaded videos can be processed by either pipeline via the web UI.

You can optionally enable person identification (attributes + re-identification embeddings) by checking the "Enable person identification" box on the upload form.