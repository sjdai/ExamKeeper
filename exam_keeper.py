import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def index():
    return render_templatr('exam.html')
