
from flask import Flask, jsonify, request ,render_template

from data_preprocessing import remove_spaces,expand_text,handling_accented,clean_data,lemmatization,join_list


import pickle


app = Flask(__name__)

tfidf_model = pickle.load(open('models/tfidf_model.pkl', 'rb'))

model = pickle.load(open('models/model.pkl', 'rb'))



@app.route('/')
def home():
    return jsonify({'response' : 'This is home !'})

@app.route('/predict', methods = (['GET','POST']))
def analyze_sentiment():

    if request.method == "POST":

        requested_data = request.get_data(as_text = True)

        clean_text_test = remove_spaces(requested_data)

        clean_text_test = expand_text(clean_text_test)

        clean_text_test = handling_accented(clean_text_test)

        clean_text_test = clean_data(clean_text_test)

        clean_text_test = lemmatization(clean_text_test)

        clean_text_test = join_list(clean_text_test)

        vector = tfidf_model.transform([clean_text_test])

        result = model.predict(vector)
       

        return render_template('index.html', sentiment=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
