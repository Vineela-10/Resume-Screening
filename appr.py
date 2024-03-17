# Importing essential libraries
from flask import Flask, render_template, request
import pickle

import re
# Load the classifier and vectorizer
classifier = pickle.load(open('svcmod.pkl', 'rb'))
cv = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Initialize Flask app and set template folder
app = Flask(__name__, template_folder='Template')



def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText
# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file-inp']
        
        # Read content from the uploaded file
        content = file.read().decode("utf-8")
        cleaned=cleanResume(content)
        # Perform prediction using the classifier
        indata = [cleaned]
        count_vect = cv.transform(indata).toarray()
        predictions = classifier.predict(count_vect)
        
        
        return render_template('predict.html', prediction=predictions)

if __name__ == '__main__':
    app.run(debug=True)
