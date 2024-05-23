from flask import Flask, Response, render_template, request, jsonify
from exercise import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/trial')
def trial():
    return render_template("landing.html")
    
@app.route('/biceps')
def biceps():
    return render_template('biceps.html')

@app.route('/triceps')
def triceps():
    return render_template('triceps.html')

@app.route('/plank')
def plank():
    return render_template('plank.html')

# Routes to stream video frames
@app.route('/biceps_video')
def biceps_video():
    return Response(biceps_processing(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/triceps_video')
def triceps_video():
    return Response(triceps_processing(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plank_video')
def plank_video():
    return Response(plank_processing(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)