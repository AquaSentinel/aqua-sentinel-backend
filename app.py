from flask import Flask

# Create a Flask application instance
app = Flask(__name__)

# Define a route and a function to handle requests to that route
@app.route('/')
def index():
    return 'Hello, World!'

# Run the application if this file is executed directly
if __name__ == '__main__':
    app.run()