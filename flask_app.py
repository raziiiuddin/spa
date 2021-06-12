import os
import tweepy as tw
from flask import Flask, flash, request, redirect, url_for, render_template
from application import get_labels
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')
app.secret_key = "9515818656"
consumer_key = 'xIPen8FSL3U5sOgtRpZ1TMnND'
consumer_secret = 'PSwWjY66eU2Oj1sYBvupaBzisyFIStfhwsFXrfRZ26ZRq4YjxV'
access_token = '1084456681617027072-vX6q2fyFmmm2ARs5jDrgT5vpizNqF5'
access_token_secret = 'J9RMfpaLB1pEJ3ecgiXvrpdGfrg9fHyMmZ7sxd3lQ3BAf'


@app.route('/analyze')
def display():
    return render_template("display.html");


@app.route('/', methods=['GET','POST'])
def get_username():
    if request.method == 'POST':
        username = request.form["username"]
        tweets = get_twitter_data(username)
        labels = get_labels(tweets)
        print(labels)
        t = 0
        for i in labels:
            if i > 0.5:
                t += 1
        score = t/len(labels)
        new_tweets = []
        for i in tweets:
            new_tweets.append(i[0])
        return render_template("display.html", score=score, tweets=new_tweets)
    return render_template("index.html")


def get_twitter_data(screen_name):
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    alltweets = []
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)
    alltweets.extend(new_tweets)
    outtweets = [[tweet.text] for tweet in alltweets]
    print(outtweets)
    return outtweets


# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             flash('Please select an image before uploading')
#             return redirect(request.url)
#         if not allowed_file(file.filename):
#             flash('Unsupported format')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             os.system('python solve_sudoku_puzzle.py --model {} --image {}{}'.format(MODEL_PATH, IMAGES_PATH, filename))
#             return redirect(url_for('uploaded_file'))
#     return render_template('upload.html')


