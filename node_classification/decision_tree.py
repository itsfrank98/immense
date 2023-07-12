from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pickle


def train_decision_tree(train_set, save_path, train_set_labels, name):
    print("Training {} decision tree".format(name))
    cls = DecisionTreeClassifier(criterion="gini")
    cls.fit(train_set, train_set_labels)
    pickle.dump(cls, open(save_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def test_decision_tree(test_set, cls: DecisionTreeClassifier):
    predictions = cls.predict(test_set)
    leaf_id = cls.apply(test_set)
    purity = 1 - cls.tree_.impurity[leaf_id]
    return predictions, purity


def load_decision_tree(path):
    print("Loading decision tree")
    return pickle.load(open(path, 'rb'))
