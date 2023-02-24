from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(train_set, train_set_labels):
    cls = DecisionTreeClassifier(criterion="gini")
    cls.fit(train_set, train_set_labels)
    return cls


def test_decision_tree(test_set, test_set_labels, cls:DecisionTreeClassifier):
    cls.predict(test_set)

