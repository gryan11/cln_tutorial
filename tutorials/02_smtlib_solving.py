#!/usr/bin/env python
# coding: utf-8

# In[20]:


from pysmt.shortcuts import read_smtlib
import z3
from glob import glob

import torch
import cln
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
import pandas as pd

from cln import Parser


# ## Sample problems:

# In[2]:


data_dir = '../data/smtlib_problems/'
smtlib_problems = glob(data_dir+'*')


# In[3]:


smtlib_problems


# In[4]:


get_ipython().run_line_magic('cat', '{smtlib_problems[0]}')


# ## Solving with z3:

# In[5]:


s = z3.Solver()


# In[6]:


s.from_file(smtlib_problems[0])


# In[7]:


s.check()


# In[8]:


s.model()


# ## Can we solve with cln?

# In[9]:


root = read_smtlib(smtlib_problems[0])


# In[10]:


root.serialize()


# Generate a model for the smt:

# In[11]:


'''
(! (((1.0 + (skoX * -1.0)) + (skoY * -1.0)) <= skoZ)) & 
(
    (! (skoZ <= 0.0)) & 
    (
        (! (skoY <= 0.0)) & 
        (! (skoX <= 0.0))
    )
)
'''

class CLNModel(torch.nn.Module):
    def __init__(self, B):
        super(CLNModel, self).__init__()
        self.B = B
        self.eps = 0.5
        
    def forward(self, x):
        B = self.B
        eps = self.eps
        
        skoX = x[0]
        skoY = x[1]
        skoZ = x[2]
        
        
        c1 = cln.neg(cln.le(((1.0 + (skoX * -1.0)) + (skoY * -1.0)) - skoZ, B, eps))
        c2 = cln.neg(cln.le(skoZ - 0.0, B, eps))
        c3 = cln.neg(cln.le(skoY - 0.0, B, eps))
        c4 = cln.neg(cln.le(skoX - 0.0, B, eps))
        
        
        ycln = cln.prod_tnorm([c1, c2, c3, c4])
        return ycln
    
model = CLNModel(B=3)

x = torch.tensor([0.0, 0.0, 0.0])
model(x)


# In[21]:


def train(x, model):
    opt = torch.optim.Adam(params=[x] + list(model.parameters()), lr=0.01)
    
    loss_trace = []
    for i in trange(100): # MORE EPOCHS
        opt.zero_grad()
        
        cln_out = model(x)
        loss = 1 - cln_out
        
        loss_trace.append(loss.item())

        loss.backward()
        opt.step()
                
    return pd.DataFrame({'loss':loss_trace})

x = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
t = train(x, model)
plt.plot(t.loss)


# Check model results:

# In[22]:


x[0].item()


# In[23]:


s = z3.Solver()
s.from_file(smtlib_problems[0])

skoX = z3.Real('skoX')
skoY = z3.Real('skoY')
skoZ = z3.Real('skoZ')
s.add(skoX == x[0].item() and skoY == x[1].item() and skoZ == x[2].item())


# In[24]:


s


# In[25]:


s.check()


# In[26]:


s.model()


# ## Solving from random initialization:
# Can we learn from random start?
# 

# In[30]:


x = torch.tensor(np.random.uniform(-10, 10, (3,)), requires_grad=True)
x


# In[31]:


model = CLNModel(B=3)

print('x before', x, '\nloss before', 1-model(x))
trace = train(x, model)
print('\nx after', x, '\nloss after', 1-model(x))


# In[32]:


s = z3.Solver()
s.from_file(smtlib_problems[0])
skoX = z3.Real('skoX')
skoY = z3.Real('skoY')
skoZ = z3.Real('skoZ')
s.add(skoX == x[0].item() and skoY == x[1].item() and skoZ == x[2].item())
s.check()


# Clearly does not work reliably... can we do better?
# - increase learning rate
# - add decay
# - make B parameter
# - go for more epochs

# In[33]:


def improved_train(x, model):
    opt = torch.optim.Adam(params=[x] + list(model.parameters()), lr=0.25)
    
    # LR DECAY
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(opt, lambda epoch: 0.99)
    
    loss_trace = []
    for i in trange(500): # MORE EPOCHS
        opt.zero_grad()
        
        cln_out = model(x)
        loss = 1 - cln_out
        
        loss_trace.append(loss.item())

        loss.backward()
        opt.step()
        scheduler.step()
                
    return pd.DataFrame({'loss':loss_trace})


# In[34]:


x = torch.tensor(np.random.uniform(-10, 10, (3,)), requires_grad=True)


B_param = torch.nn.Parameter(torch.tensor(1.0))


model = CLNModel(B=B_param)

print('x before', x, '\nloss before', 1-model(x))
trace = improved_train(x, model)
print('\nx after', x, '\nloss after', 1-model(x))


# check if solving reliably...

# In[35]:


s = z3.Solver()
s.from_file(smtlib_problems[0])
skoX = z3.Real('skoX')
skoY = z3.Real('skoY')
skoZ = z3.Real('skoZ')
s.add(skoX == x[0].item() and skoY == x[1].item() and skoZ == x[2].item())
s.check()


# See if you can solve problems 1-3!

# In[36]:


smtlib_problems


# In[ ]:




