import json
import plotly
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Scatter
import plotly.graph_objs as go
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def count_messages_by_category(df):
    '''
    Counts the number of messages in each category
    
    Args:
    df: dataframe containing messages and respective classification into categories
    
    Returns:
    category_list: list of categories
    count_messages: total number of messages grouped by category
    '''
    count_df = df.iloc[:,4:].sum().sort_values(ascending=False)
    category_list = list(count_df.index)
    count_messages = list(count_df.values)
    
    return category_list, count_messages

def message_proportion(df):
    '''
    Calculates the proportion of messages in each category
    
    Args: 
    df: dataframe containing messages and respective classification into categories
    
    Returns: 
    category_list: list of categories
    percent_messages: percentage of messages by category
    '''
    total_num_messages = df.shape[0]
    percent_df = round(df.iloc[:,4:].sum().sort_values(ascending=False)/total_num_messages * 100 , 2 )
    category_list = list(percent_df.index)
    percent_messages = list(percent_df.values)
    
    return category_list, percent_messages

def text_to_plot (text):
    '''
    Joins all messages into one single and continue text.
    
    Args:
    text: list of messages to extract text
    
    Returns:
    words: list with the all the words from text, cleaned and lemmatized
    '''
    words=",".join(text)
    words = re.sub(r'[^a-zA-Z0-9]',' ' , words.lower())
    words = WordNetLemmatizer().lemmatize(words)
    
    return words


def plotly_wordcloud(text):
    '''
    Function that prepares the text to be plotted as a word cloud, based on the frequency of each word
    
    Args:
    text: str - text to be splitted into words and represented in a word cloud
    
    Returns:
    data: list of words and respective frquency, prepared to be plotted as a Scatter
  
    '''
    
    wc = WordCloud(
        stopwords = set(STOPWORDS),
        prefer_horizontal = 0.5,
        collocations = False,
        max_font_size = 15,
        max_words = 150,
        background_color='white').generate(text)
    
    word_list=[]
    freq_list=[]
    fontsize_list=[]
    position_list=[]
    orientation_list=[]
    color_list=[]

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)
        
    # get the positions
    x=[]
    y=[]
    for i in position_list:
        x.append(i[0])
        y.append(i[1])
            
    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i*100)
    new_freq_list
    
    data = Scatter(x=x,
                   y=y, 
                   textfont = dict(size=new_freq_list,color=color_list),
                   #hoverinfo='text',
                   #hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                   mode='text',  
                   text=word_list
                  )
    
    return data

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('CleanDisasterMessages', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Genre Counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Categories Counts
    category_list, count_messages = count_messages_by_category(df)
    
    # Categories Proportion
    category_list, percent_messages = message_proportion(df)
    
    # Words Cloud
    words = text_to_plot(df.message)
    data = plotly_wordcloud(words)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    graphs = [
         {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
         {
            'data': [
                Bar(
                    x=category_list,
                    y=count_messages
                )
            ],

            'layout': {
                'title': 'Number of Messages by Category',
                'yaxis': {
                    'title': "Count of messages"
                },
                'xaxis': {
                    'title': "Disaster Categories"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=category_list[:10],
                    values=percent_messages[:10]
                )
            ],

            'layout': {
                'title': 'Top 10 Categories with most messages',
                'yaxis': {
                    'title': "Percentage of messages"
                },
                'xaxis': {
                    'title': "Disaster Categories",
                    'tickangle': 45
                }
            }
        },
        {
            'data': [data],

            'layout': {
                'xaxis': {'visible': False ,'showgrid': False, 'showticklabels': False, 'zeroline': False},
                'yaxis': {'visible': False ,'showgrid': False, 'showticklabels': False, 'zeroline': False},
                'height': 700,
                'line': {'shape': 'linear'}
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()