class CountVectorizer:
    """
    Конвертирует слова в вектора

    """
    def get_feature_names(self) -> list:
        return list(self.unique_words)

    def fit_transform(self, corpus: list) -> list:
        if not isinstance(corpus, (str, list)):
            raise TypeError('Incorrect type of the input data')

        self.unique_words = set()
        for string in corpus:
            list_split = string.split()
            for word in list_split:
                self.unique_words.add(word.lower())

        token2id = {token: i for i, token in enumerate(self.unique_words)}
        list_embedding = []
        for i in range(len(corpus)):
            list_embedding.append([0 for num in range(len(token2id))])

        for i in range(len(corpus)):
            sentence_list = corpus[i].split()
            for word in sentence_list:
                list_embedding[i][token2id[word.lower()]] += 1
        return list_embedding


if __name__ == '__main__':
    corpus = ['Crock Pot Pasta Never boil pasta again',
              'Pasta Pomodoro Fresh ingredients Parmesan to taste']
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(count_matrix)
    print(vectorizer.get_feature_names())
