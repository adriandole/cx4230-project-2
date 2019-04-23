import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# observations taken at the intersection of 5th and Spring St.
x = np.array([1, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3,
              1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 1, 2]).reshape((-1, 1))
y = np.array(
    [4.02, 9, 3.47, 4.18, 7.58, 11.42, 8.49, 10.91, 14.21, 12.22, 14.45, 16.74, 4.34, 7.21, 10.54, 4.79, 8.52, 11.68,
     14.92, 6.12, 8.81, 12.27, 14.63, 4.01, 6.08, 8, 9.73, 5.95, 10.1, 13.52, 3.89, 6.56, 9.31, 11.56, 9.39, 11.45,
     13.58, 16.37, 20.34, 4.16, 7, 8.89, 4.41, 6.48])

lr = linear_model.LinearRegression()
lr.fit(x, y)
c1 = lr.coef_
c2 = lr.intercept_

plt.scatter(x, y)
plt.plot([1, 5], [c1 + c2, 5*c1 + c2], color='red')
plt.xlabel('# of cars in front')
plt.ylabel('Wait time (s)')

eqn = "y = {:2f}n + {:2f}".format(c1[0], c2)
plt.text(3, 19, eqn)

print(eqn)
plt.show()
