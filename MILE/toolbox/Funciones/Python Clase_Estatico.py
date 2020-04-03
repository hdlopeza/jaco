#%%
class Metodo():
    name = "Hernan"
    na =''
    def instanceMethod(self):
        self.lastname = "Lopez"
        print(self.name)
        print(self.lastname)

    @classmethod
    def classMethod(cls):
        cls.name = "Archila"
        print(cls.name)

    @staticmethod
    def staticMethod(var):
        na = var
        print(var)

#%%
Metodo.staticMethod('var1')

#%%
Metodo.classMethod()

#%%
# Creates an instance of the class
a = Metodo()

#%%
# Calls instance method
a.instanceMethod()

#%%
a.name

#%%
a.lastname

#%%
a.staticMethod('var2')

#%%
a.na

#%%
