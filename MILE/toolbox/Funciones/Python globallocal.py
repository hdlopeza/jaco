#%%
x = 'global'


#%%
def pp():
    x = 'otro'
    print x
pp()
print x

#%%
def otro():
    global x 
    x = 'cambiado'
    print x
otro()
print x


#%%
