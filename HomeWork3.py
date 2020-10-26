from CountVectorizer import CountVectorizer
if __name__ == '__main__':
    corpus = ['Crock Pot Pasta Never boil pasta again',
              'Pasta Pomodoro Fresh ingredients Parmesan to taste']
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(count_matrix)
    print(vectorizer.get_feature_names())
