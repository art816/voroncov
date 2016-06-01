import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data[data['Sex'] == 'female'].count()
data[data['Sex'] == 'male'].count()
data[data['Survived'] == 1].Embarked.count()
fm_p = data[data['Sex'] == 'female']
a = lambda x: x.split()[x.split().index('Mrs.') + 1] if 'Mrs.' in x else x
fm_p.Name.apply(a)

