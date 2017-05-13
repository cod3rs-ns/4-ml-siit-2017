from em import ExpectationMaximization

if __name__ == "__main__":
    clusters = ['a', 'b', 'c', 'd']

    data = [
        [1, 2, 3],
        [1, 2, 3],
        [2, 3, 4]
    ]

    em = ExpectationMaximization(clusters)
    em.fit(data)
    em.predict()
