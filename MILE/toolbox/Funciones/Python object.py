class prueba:
    def __init__(self, numero):
        self.numero = numero

    def to2(self):
        self.numero = self.numero*2


n = prueba(2)
n.to2()
print(n.numero)

p00 = [[791,42],[877,42],[877,76],[791,76]]

class prueba1:
    def __init__(self, punto):
        self.punto = punto

n1 =  prueba1(p00)
print(n1.punto)

#Como funciona el superinit?