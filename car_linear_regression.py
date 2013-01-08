# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from datetime import datetime
from datetime import timedelta
from sklearn.linear_model import LinearRegression

# <markdowncell>

# Begin by defining X (our features) which represent the number of miles driven.

# <codecell>

X = [[10], [130], [1659], [3279], [3332], [3386], [3534]]

# <markdowncell>

#  Then we define y (our labels) which represent the number of days since my purchase date.

# <codecell>

y = [0, 5, 35, 66, 69, 70, 73]

# <codecell>

figsize(10.2, 5.1)
scatter(X, y, color='black')
pylab.ylim([0, (int(max(y)) * 1.05) + 1])
pylab.xlim([0, (max(X)[0] * 1.05)])
ax=pylab.gca()
ax.yaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
pylab.axes().set_xlabel('Miles')
pylab.axes().set_ylabel('Time')
pylab.axes().set_title('Odometer Readings over time')

# <codecell>

purchase_date = datetime(2012, 10, 26)

# <codecell>

clf = LinearRegression()
# fit the classifier to our dataset
clf.fit(X, y)

# Use our classifier to predict the dates of future events
# Complimentary maintenance occurs ever 5000 miles up to 25,000 miles.
X_pred = [[5000], [10000], [15000], [20000], [25000]]
days = clf.predict(X_pred)
for data in zip(X_pred, days):
    date = purchase_date + timedelta(int(round(data[1], 0)))
    print '{0} miles on {1}'.format(data[0][0], date.strftime('%b %d %Y'))

# <codecell>

def plot_data(X, y, X_pred, clf, title):
    """A function used to plot the data in this example"""
    
    figsize(10.2, 5.1)
    X2 = [[x] for x in linspace(0, max(X_pred)[0] * 1.15, 50)]
    days = clf.predict(X_pred)
    scatter(X, y, color='black')
    scatter(X_pred, days, color='red')
    plot(X2, clf.predict(X2), color='blue')
    
    i = 0
    len_data = len(X_pred)
    xytext = (5, -10)
    for data in zip(X_pred, days):
        i = i + 1
        dat = purchase_date + timedelta(int(round(data[1], 0)))

        annotate(dat.strftime('%b %d %Y'), xy=(data[0][0], data[1]), xycoords='data', xytext=xytext, textcoords='offset points')
        
    pylab.ylim([0, int(clf.predict(X2).max()) + 1])
    pylab.xlim([0, max(X2)[0]])
    ax=pylab.gca()
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    pylab.axes().set_xlabel('Miles')
    pylab.axes().set_ylabel('Time')
    pylab.axes().set_title(title)

# <codecell>

X_pred = X_pred = [[5000]]
plot_data(X, y, X_pred, clf, 'Projected date of 1st Complimentary Maintenance')

# <codecell>

X_pred = [[5000], [10000], [15000], [20000], [25000]]
plot_data(X, y, X_pred, clf, 'Projected Dates of All Complimentary Maintenance')

# <codecell>

X_pred = [[36000], [50000], [60000], [70000], [80000], [100000]]
plot_data(X, y, X_pred, clf, 'Projected Dates of All Warranty Expirations')
