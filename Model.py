import numpy as np
import pandas as pd
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense

# User Input Parameters
m = int(input("Enter the number of sites (m): "))
n = int(input("Enter the number of warehouse (n): "))
noScen = 2
warehouses = ['Warehouse ' + str(i) for i in range(1, n+1)]
scenarios = ['Scenario ' + str(k) for k in range(1, noScen+1)]

# Generate Datas
cost_l = np.random.randint(50, 200, n)
cost_q = np.random.randint(100, 500, n)
cost_b = np.random.randint(0, 20, m)
cost_s = np.random.randint(0, 10, m)
req = np.random.randint(0, 10, n * m)
needs = np.random.randint(0, 15, size=(n * noScen))

# Reshape Datas
req_reshaped = req.reshape((n, m))
needs_reshaped = needs.reshape((n, noScen))

# Condition Check (sj < bj)
for j in range(m):
    if cost_s[j] >= cost_b[j]:
        cost_s[j] = cost_b[j] - 10

# Create Model
problem1 = Container()

# Define Sets
i = Set(container=problem1, name='i', description='Warehouses', records=['Warehouse ' + str(i) for i in range(1, n+1)])
j = Set(container=problem1, name='j', description='Sites', records=['Site ' + str(j) for j in range(1, m+1)])
k = Set(container=problem1, name='k', description='Scenarios', records=['Scenario ' + str(k) for k in range(1, noScen+1)])

# Set Parameters
d = Parameter(container=problem1, name='d', domain=[i, k], description='Products needs', records=needs_reshaped)
b = Parameter(container=problem1, name='b', domain=[j], description='Preorder cost', records=cost_b)
l = Parameter(container=problem1, name='l', domain=[i], description='Additional cost', records=cost_l)
q = Parameter(container=problem1, name='q', domain=[i], description='Selling price', records=cost_q)
s = Parameter(container=problem1, name='s', domain=[j], description='Salvage values', records=cost_s)
c = Parameter(container=problem1, name='c', domain=[i], description='Cost coefficients', records=cost_l-cost_q)
prob = Parameter(container=problem1, name='p', domain=[k], description='Scenario probabilities', records=np.array([1 / 2, 1 / 2]))
A = Parameter(container=problem1, name='A', description='Requirements of products', domain=[i, j], records=req_reshaped)

# Set Variables
x = Variable(container=problem1, name='x', type='positive', domain=j, description='Number of parts to be ordered before production')
y = Variable(container=problem1, name='y', type='positive', domain=[j, k], description='Number of parts left in inventory')
z = Variable(container=problem1, name='z', type='positive', domain=[i, k], description='Number of units produced')

# Build Equations
need_constraint = Equation(container=problem1, name='need_constraint', domain=[j, k])
need_constraint[j, k] = y[j, k] == x[j] - Sum(i, A[i, j] * z[i, k])

non_negative_order_constraint = Equation(container=problem1, name='non_negative_order_constraint', domain=j)
non_negative_order_constraint[j] = x[j] >= 0

non_negative_inventory_constraint = Equation(container=problem1, name='non_negative_inventory_constraint', domain=[j, k])
non_negative_inventory_constraint[j, k] = y[j, k] >= 0

non_negative_production_constraint = Equation(container=problem1, name='non_negative_production_constraint', domain=[i, k])
non_negative_production_constraint[i, k] = z[i, k] >= 0

production_need_constraint = Equation(container=problem1, name='production_need_constraint', domain=[i, k])
production_need_constraint[i, k] = z[i, k] <= d[i, k]

inventory_salvage_constraint = Equation(container=problem1, name='inventory_salvage_constraint', domain=[j, k])
inventory_salvage_constraint[j, k] = y[j, k] >= 0

production_cost = Sum(i, c[i] * z[i, k])
salvage_value = Sum(j, s[j] * y[j, k])
objective_expression = production_cost - salvage_value
objective_func = Sum(j, b[j] * x[j]) + Sum(k, objective_expression * prob[k])
# GAMS Model
problem1 = Model(container=problem1, name='problem_1', equations=problem1.getEquations(), problem='MIP', sense=Sense.MIN, objective=objective_func)

# Solve Model
problem1.solve()

# Print Results
print("Objective Value:", problem1.objective_value)
print("The number of units produced:")
print(z.records.set_index(["i", "k"]))