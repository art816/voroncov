import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data[data['Sex'] == 'female'].count()
data[data['Sex'] == 'male'].count()
data[data['Survived'] == 1].Embarked.count()
fm_p = data[data['Sex'] == 'female']
a = lambda x: x.split()[x.split().index('Mrs.') + 1] if 'Mrs.' in x else x
fm_p.Name.apply(a)



import numpy
sclearn_data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
sex_dict = {'male': 0, 'female': 1}
filter_scl_data = sclearn_data
filter_scl_data['Sex'] = filter_scl_data['Sex'].apply(sex_dict.get).astype(int)
for column in ['Pclass', 'Fare', 'Age', 'Sex']:
    filter_scl_data = filter_scl_data[numpy.isnan(sclearn_data[column]) == False]


Bull = filter_scl_data.Survived
sclearn_data = filter_scl_data[['Pclass', 'Fare', 'Age', 'Sex']]

from sklearn.tree import DecisionTreeClassifier
cls = DecisionTreeClassifier(random_state=241)
cls.fit(sclearn_data, Bull)
dict_key_importance = dict(zip(['Pclass', 'Fare', 'Age', 'Sex'], cls.feature_importances_))





