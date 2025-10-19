## 1. Prerequisites:

Ensure you have Python installed on your system. Python 3 is recommended.

## 2. Install Flask:

It is best practice to install Flask within a virtual environment to manage dependencies for your project in isolation.

```
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Required packages :

Install dependencies from requirements.txt:

```
pip install -r requirements.txt
```

## 4. Generate requirements (optional)

```
pip freeze > requirements.txt
```

## 5. Make folder called models at the root of repository and insert the files downloaded from the link (https://drive.google.com/drive/folders/1uWs1QxCHI5M8N5F8j4dIPBJi3lyQkk-0?usp=drive_link) into this models folder


## 6. Run the Application:

```
python app.py
```

Applicationn starts running on http://127.0.0.1:5000

## 7. To test /detect route

Corrected curl (use single leading slashes in local absolute paths):

```
curl -X POST http://127.0.0.1:5000/detect \
  -F "ship=@/home/tanishq/Desktop/aqua-sentinel-backend/inference/images/ship/000006_jpg.rf.6e6d1d62bdbe00c3061830c65f0233d7.jpg" \
  -F "debris=@/home/tanishq/Desktop/aqua-sentinel-backend/inference/images/debris/20170227_203601_0c46_16903-29838-16.jpg" \
  -o detections.zip
```

After download, extract and view:
```
unzip detections.zip
# then open ship_output.jpg / debris_output.jpg with your image viewer
```

# Routes 

### /detect
"""
    Accepts multipart/form-data with two files:
      - preferred form field names: 'ship' and 'debris'
      - or any two uploaded files (first -> ship, second -> debris)

    Returns a ZIP file containing:
      - ship_output.jpg
      - debris_output.jpg
"""

