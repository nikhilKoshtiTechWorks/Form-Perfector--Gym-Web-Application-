from flask import Flask, Response, render_template, request, jsonify, flash, redirect, url_for, make_response, session
# import jwt
from datetime import datetime, timedelta
from functools import wraps
from exercise import *

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# the above key is generated using os.urandom(12) function in python shell
app.secret_key = "k\xab\x86;y}\xf1E\xe7\xac~\x11"            # used for encryption and decryption
connection = psycopg2.connect(os.environ["DATABASE_URL"])

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/register', methods = ['GET','POST'])
def register():
    if request.method == 'POST':
        return redirect(url_for('/'))
    
    return render_template('register.html')

@app.route('/login', methods = ['GET','POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('/'))
    
    return render_template('login.html')

# Route to render the exercise page
@app.route('/exercise/<exercise_name>')          # getting the exercise_name from the index.html
def exercise(exercise_name):
    # video = exercise_name
    return render_template('exercise_template.html', video = exercise_name)

@app.route('/insert', methods = ["POST", "GET"])
def insert():
    if request.method == "POST":
        # Assuming you have the following variables from your form
        exercise_name = request.form["exerciseName"]
        targeted_angles = int(request.form["angleCount"])
        min_values = []
        max_values = []
        angle_names = []  # This should be populated with the names of the angles

        for i in range(targeted_angles):
            min_value = request.form[f"minValue{i}"]
            max_value = request.form[f"maxValue{i}"]
            angle_name = request.form[f"angleName{i}"]  # Get the angle name
            min_values.append(min_value)
            max_values.append(max_value)
            angle_names.append(angle_name)

        # Construct column names for the SQL query
        columns = []
        values = []

        for angle in angle_names:
            min_col = f"min_{angle.replace(' ', '_').lower()}"
            max_col = f"max_{angle.replace(' ', '_').lower()}"
            columns.append(min_col)
            columns.append(max_col)

        for i in range(targeted_angles):
            values.append(min_values[i])
            values.append(max_values[i])

        # Construct the full SQL insert statement
        column_string = ", ".join(["exercise_name"] + columns)
        print(column_string)
        value_placeholders = ", ".join(["%s"] + ["%s"] * (2 * targeted_angles))     # this is a place holder we will pass the values in .execute()
        print(value_placeholders)
        sql_query = f"INSERT INTO exercises ({column_string}) VALUES ({value_placeholders}) RETURNING eid"
        print(sql_query)

        with connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_query, [exercise_name] + values)     
                # exercise_name & values goes in a value_placeholders where %s is present respectively

        connection.commit()
        cursor.close()

        flash('Exercise details inserted successfully!', 'success')
        return redirect(url_for('insert'))


    else:
        return render_template('insert.html')

# Routes to stream video frames
@app.route('/video_stream/<exercise_video>')
def exercise_video(exercise_video):
    sql_query = f"SELECT * FROM exercises WHERE exercise_name = '{exercise_video}'"
    
    with connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_query)
                result = cursor.fetchall()
                
                # arr = [0] * 24

                # for i in range(2, len(result[0])):
  
                #     # Converts each element to an integer
                #     arr[i-2] = int(result[0][i])
                
    return Response(exercise_processing(result), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)