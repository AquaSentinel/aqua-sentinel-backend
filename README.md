## 1. Prerequisites:

Ensure you have Python installed on your system. Python 3 is recommended.

## 2. Install Flask:

It is best practice to install Flask within a virtual environment to manage dependencies for your project in isolation.

```
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

## 3. Install Required packages :

Install dependencies from requirements.txt: If you receive a project with a requirements.txt file, you can install all its dependencies by activating your virtual environment and running:

```
    pip install -r requirements.txt
```

```
    pip install package_name
```

## 4. To add packages installed in requirements.txt : 

Generate requirements.txt: Once all dependencies are installed in your virtual environment, you can generate the requirements.txt file with the exact versions of all installed packages.

```
    pip freeze > requirements.txt
```

## 5. Run the Application:

```
python app.py
```