import numpy as np
import matplotlib.pyplot as plt


########LINEAL#######


x = np.arange(-5.0, 5.0, 0.1)

##Se puede ajustar la pendiente y la intersección para verificar los cambios en el gráfico

y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Variable dependiente')
plt.xlabel('Variable indepdiendente')
plt.show()

######CUADRATICA######

x = np.arange(-5.0, 5.0, 0.1)

##Se puede ajustar la pendiente y la intersección para verificar los cambios en el gráfico
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r')
plt.ylabel('Variable dependiente')
plt.xlabel('Variable independiente')
plt.show()

#######EXPONENCIAL#######

X = np.arange(-5.0, 5.0, 0.1)

##Se puede ajustar la pendiente y la intersección para verificar los cambios en el gráfico

Y= np.exp(X)

plt.plot(X,Y)
plt.ylabel('Variable Dependiente')
plt.xlabel('Variable Independiente')
plt.show()


########LOGARITMICO########

X = np.arange(-5.0, 5.0, 0.1)

Y = np.log(X)

plt.plot(X,Y)
plt.ylabel('Variable Dependiente')
plt.xlabel('Variable Independiente')
plt.show()


########SIGMOIDAL/LOGÏSTICA##########


X = np.arange(-5.0, 5.0, 0.1)


Y = 1-4/(1+np.power(3, X-2))

plt.plot(X,Y)
plt.ylabel('Variable Dependiente')
plt.xlabel('Variable Independiente')
plt.show()


