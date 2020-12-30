from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

app = Flask(__name__)
    
@app.route('/', methods=['GET', 'POST'])

def index():

    if request.method == 'POST':
        
        output = ""
        tags = ""
        
        #Read data
        df = pd.read_csv('tags.csv')
        df = df.drop(['Unnamed: 0'], axis = 1)
        
        import ast
        df['Tags'] = df['Tags'].apply(lambda x: ast.literal_eval(x))

        tfidf = TfidfVectorizer(analyzer = 'word' , max_features=10000, ngram_range=(1,3) , stop_words='english')
        X = tfidf.fit_transform(df['Title'])

        multilabel = MultiLabelBinarizer()
        y = df['Tags']
        y = multilabel.fit_transform(y)
        
        X_train, X_test, y_train , y_test = train_test_split(X , y, test_size = 0.2, random_state= 0)
        sgd = SGDClassifier()
        clf_sgd = OneVsRestClassifier(sgd)
        clf_sgd.fit(X_train, y_train)

        #reading movie title given by user in the front-end
        Question = request.form.get('fquestion')
        print(Question)
       
        def getTags(question):
            question = [question]
            question = tfidf.transform(question)
            tags = multilabel.inverse_transform(clf_sgd.predict(question))
            return tags
        
        try:
            out = getTags(Question)
            out1 = [item for t in out for item in t] 
            out1 = map(lambda e : "<" + e + ">", out1) 
            str1 = ","
            output = str1.join(out1)  
            return render_template('index.html', tags=output, question=Question)
        except ValueError as e:
            print("error")
            return render_template('index.html', error=e)

    else:
        return render_template('index.html')
    
if __name__ == '__main__':
   app.run(debug = True)
