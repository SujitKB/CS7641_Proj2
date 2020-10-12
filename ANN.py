
import pandas as pd
import numpy as np
import time as t
import matplotlib.pyplot as plt
import time
import mlrose_hiive as mlr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

from preprocess_data import breast_cancer_diagnostic

RANDOM_STATE = 42

X, Y = breast_cancer_diagnostic()

# Split data into Train and Test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=RANDOM_STATE, stratify=Y)
train_sizes = np.linspace(0.01, 1.0, 5)

############################### Gradient Descent
# Instantiate ANN
lr_list=[]
accuracy_list=[]
for lr in np.arange(0.001, 0.1, 0.0050):
    ann = mlr.NeuralNetwork(hidden_nodes=[4,3], activation='relu',
                                algorithm='gradient_descent', 
                                max_iters=5000, max_attempts=100,
                                bias=True, learning_rate=lr,
                                early_stopping=True, random_state=RANDOM_STATE, curve=True)

    lr_list.append(lr)

    ann.fit(X_train, Y_train)

    Y_pred = ann.predict(X_test)
    accuracy_ann = accuracy_score(Y_test, Y_pred) * 100
    accuracy_list.append(accuracy_ann)
    print('Accuracy Score of ANN with gradient_descent is %.2f%% with lr = %.4f' % (accuracy_ann, lr))

print(lr_list, accuracy_list)
plt.figure()
plt.plot(lr_list, accuracy_list, marker='o', label='Accuracy Score')
plt.title('Accuracy Score v/s Learning Rate [Gradient Descent]')
plt.xlabel('Learning Rate')
plt.ylabel("Accuracy Score (Test Dataset)")
plt.legend()
plt.grid()
plt.savefig('ann_grad_descent.png')
plt.clf()

############################## Simulated Annealing
# Instantiate ANN
lr=0.02
t_list=[]
accuracy_list=[]

t=0.001
while t <= 100 : #in np.arange(0.001, 10, 0.005):
    schedule=mlr.ExpDecay(init_temp=t)
    ann = mlr.NeuralNetwork(hidden_nodes=[4,3], activation='relu',
                                algorithm='simulated_annealing',
                                schedule=schedule,
                                max_iters=5000, max_attempts=200,
                                bias=True, learning_rate=lr,
                                early_stopping=True, random_state=RANDOM_STATE, curve=True)

    t_list.append(t)

    ann.fit(X_train, Y_train)

    Y_pred = ann.predict(X_test)
    accuracy_ann = accuracy_score(Y_test, Y_pred) * 100
    accuracy_list.append(accuracy_ann)

    print('Accuracy Score of ANN with simulated_annealing is %.2f%% with t = %.4f' % (accuracy_ann, t))
    t *= 10

print(t_list, accuracy_list)
plt.figure()
plt.plot(t_list, accuracy_list, marker='o', label='Accuracy Score')
plt.title('Accuracy Score v/s Initial Temperature [Simulated Annealing]')
plt.xlabel('Initial Temperature')
plt.ylabel("Accuracy Score (Test Dataset)")
plt.legend()
plt.grid()
plt.savefig('ann_sim_anealing.png')
plt.clf()


###################################### random_hill_climb
# Instantiate ANN
restart_list=[]
accuracy_list=[]
lr=0.02
for restart in np.arange(0, 10, 1):
    ann = mlr.NeuralNetwork(hidden_nodes=[4,3], activation='relu',
                                algorithm='random_hill_climb', restarts=restart,
                                max_iters=5000,max_attempts=200,
                                bias=True, learning_rate=lr,
                                early_stopping=True, random_state=RANDOM_STATE, curve=True)

    restart_list.append(restart)

    ann.fit(X_train, Y_train)

    Y_pred = ann.predict(X_test)
    accuracy_ann = accuracy_score(Y_test, Y_pred) * 100
    accuracy_list.append(accuracy_ann)
    print('Accuracy Score of ANN with random_hill_climb is %.2f%% with restart# = %.4f' % (accuracy_ann, restart))

print(restart_list, accuracy_list)
plt.figure()
plt.plot(restart_list, accuracy_list, marker='o', label='Accuracy Score')
plt.title('Accuracy Score v/s Restart # [Random Hill Climb - Restarts]')
plt.xlabel('Restart #')
plt.ylabel("Accuracy Score (Test Dataset)")
plt.legend()
plt.grid()
plt.savefig('ann_rhc.png')
plt.clf()


###################################### Genetic Algorithm
# Instantiate ANN
lr_list=[]
accuracy_list=[]
mut_prob=0
lr=0.02


while mut_prob < 0.5:
    mut_prob += 0.1
    pop_size = 0
    while pop_size < 500:
        pop_size += 100
        ann = mlr.NeuralNetwork(hidden_nodes=[4,3], activation='relu',
                            algorithm='genetic_alg',
                            pop_size=pop_size,mutation_prob=mut_prob,
                            max_iters=5000,max_attempts=200,
                            bias=True, learning_rate=lr,
                            early_stopping=True, random_state=RANDOM_STATE, curve=True)

        lr_list.append(str("{:.1f}".format(mut_prob)) + ' & ' + str(pop_size))

        ann.fit(X_train, Y_train)

        Y_pred = ann.predict(X_test)
        accuracy_ann = accuracy_score(Y_test, Y_pred) * 100
        accuracy_list.append(accuracy_ann)
        print('Accuracy Score of ANN with genetic_alg is %.2f%% with population = %.4f and mut_prob = %.4f' % (accuracy_ann, pop_size, mut_prob))



print(lr_list, accuracy_list)
plt.figure(figsize=(20,18))
plt.rcParams.update({'font.size':20})
plt.rcParams['axes.labelweight'] = 'bold'
plt.xticks(rotation=90)
plt.plot(lr_list, accuracy_list, marker='o', label='Accuracy Score')
plt.title('Accuracy Score v/s (mutation_prob & pop_size [Genetic Algorithm]')
plt.xlabel('mutation_prob & pop_size')
plt.ylabel("Accuracy Score (Test Dataset)")
plt.legend()
plt.grid(True)
plt.savefig('ann_ga.png')
plt.clf()
#print('Accuracy Score of ANN with simulated_annealing is %.2f%% with lr = %.3f' % (accuracy_ann,lr))


#############################################################Train Time & Accuracy Comparison
# Algorithm Comparison metrics
algo_accuracy   = [0.] * 4
algo_train_time = [0.] * 4
algo_query_time = [0.] * 4

max_attempt=200
lr=0.02

####### Gradient Descent
ann = mlr.NeuralNetwork(hidden_nodes=[4,3], activation='relu',
                                algorithm='gradient_descent',
                                max_iters=5000, max_attempts=max_attempt,
                                bias=True, learning_rate=lr,
                                early_stopping=True, random_state=RANDOM_STATE, curve=True)
i=0
tStart = time.time()
ann.fit(X_train, Y_train)
tEnd = time.time()
algo_train_time[i] = tEnd - tStart

Y_pred = ann.predict(X_test)

accuracy_ann = accuracy_score(Y_test, Y_pred) * 100
algo_accuracy[i] = accuracy_ann

####### Random Hill Climbing (restarts)
ann = mlr.NeuralNetwork(hidden_nodes=[4, 3], activation='relu',
                        algorithm='random_hill_climb', restarts=8,
                        max_iters=5000, max_attempts=max_attempt,
                        bias=True, learning_rate=lr,
                        early_stopping=True, random_state=RANDOM_STATE, curve=True)

i=1
tStart = time.time()
ann.fit(X_train, Y_train)
tEnd = time.time()
algo_train_time[i] = tEnd - tStart

Y_pred = ann.predict(X_test)

accuracy_ann = accuracy_score(Y_test, Y_pred) * 100
algo_accuracy[i] = accuracy_ann

####### Simulated Annealing
schedule = mlr.ExpDecay(init_temp=0.001)
ann = mlr.NeuralNetwork(hidden_nodes=[4, 3], activation='relu',
                        algorithm='simulated_annealing',
                        schedule=schedule,
                        max_iters=5000, max_attempts=max_attempt,
                        bias=True, learning_rate=lr,
                        early_stopping=True, random_state=RANDOM_STATE, curve=True)

i=2
tStart = time.time()
ann.fit(X_train, Y_train)
tEnd = time.time()
algo_train_time[i] = tEnd - tStart

Y_pred = ann.predict(X_test)

accuracy_ann = accuracy_score(Y_test, Y_pred) * 100
algo_accuracy[i] = accuracy_ann

####### Genetic Algorithm
ann = mlr.NeuralNetwork(hidden_nodes=[4, 3], activation='relu',
                        algorithm='genetic_alg',
                        pop_size=200, mutation_prob=0.1,
                        max_iters=5000, max_attempts=max_attempt,
                        bias=True, learning_rate=lr,
                        early_stopping=True, random_state=RANDOM_STATE, curve=True)

i=3
tStart = time.time()
ann.fit(X_train, Y_train)
tEnd = time.time()
algo_train_time[i] = tEnd - tStart

Y_pred = ann.predict(X_test)

accuracy_ann = accuracy_score(Y_test, Y_pred) * 100
algo_accuracy[i] = accuracy_ann

################ Comparison ###################
learners = ('Gradient Descent', 'RHC (8-Restart)', 'SA', 'GA')
y_pos = np.arange(len(learners))

## Accuracy Comparison Plot
plt.figure(figsize=(15,15))
plt.rcParams.update({'font.size':20})
plt.rcParams['axes.labelweight'] = 'bold'
print(algo_accuracy)
print(algo_train_time)
plt.bar(y_pos, algo_accuracy, width=0.5)
plt.xticks(y_pos, learners)
plt.grid(True)
plt.ylim((80, 98))
plt.title('Comparison of Maximum Accuracy Score')
plt.ylabel('Accuracy')
plt.savefig('ann_Accuracy_Classifiers.png')
plt.clf()

# ####################Training Time Comparison Plot
plt.figure(figsize=(15,15))
plt.rcParams.update({'font.size':20})
plt.rcParams['axes.labelweight'] = 'bold'

plt.bar(y_pos, algo_train_time, width=0.5)
plt.xticks(y_pos, learners)
plt.grid(True)
plt.ylim((0, 240))
plt.title('Comparison of Training Time')
plt.ylabel('Train Time (sec)')
plt.savefig('ann_TrainTime.png')
plt.clf()


#############################################################Training Loss Comparison
train_losses_gd=[]
train_losses_rhc=[]
train_losses_sa=[]
train_losses_ga=[]

# For each max iterations to run for
max_iter_lst=np.arange(0,5000,50)
max_attempt = 200
lr = 0.02
for max_iter in range(0,5000,50):

    ####### Gradient Descent
    ann = mlr.NeuralNetwork(hidden_nodes=[4,3], activation='relu',
                                algorithm='gradient_descent',
                                max_iters=max_iter, max_attempts=max_attempt,
                                bias=True, learning_rate=lr,
                                early_stopping=True, random_state=RANDOM_STATE, curve=True)

    ann.fit(X_train, Y_train)
    train_loss_gd = log_loss(Y_train, ann.predict(X_train))
    train_losses_gd.append(train_loss_gd)

    #Y_pred = ann.predict(X_test)
    #accuracy_ann = accuracy_score(Y_test, Y_pred) * 100

    ####### Random Hill Climbing (restarts)
    ann = mlr.NeuralNetwork(hidden_nodes=[4, 3], activation='relu',
                        algorithm='random_hill_climb', restarts=8,
                        max_iters=max_iter, max_attempts=max_attempt,
                        bias=True, learning_rate=lr,
                        early_stopping=True, random_state=RANDOM_STATE, curve=True)

    ann.fit(X_train, Y_train)
    train_loss_rhc = log_loss(Y_train, ann.predict(X_train))
    train_losses_rhc.append(train_loss_rhc)
    #Y_pred = ann.predict(X_test)
    #accuracy_ann = accuracy_score(Y_test, Y_pred) * 100

    ####### Simulated Annealing
    schedule = mlr.ExpDecay(init_temp=0.001)
    ann = mlr.NeuralNetwork(hidden_nodes=[4, 3], activation='relu',
                        algorithm='simulated_annealing',
                        schedule=schedule,
                        max_iters=max_iter, max_attempts=max_attempt,
                        bias=True, learning_rate=lr,
                        early_stopping=True, random_state=RANDOM_STATE, curve=True)

    ann.fit(X_train, Y_train)
    train_loss_sa = log_loss(Y_train, ann.predict(X_train))
    train_losses_sa.append(train_loss_sa)
    #Y_pred = ann.predict(X_test)
    #accuracy_ann = accuracy_score(Y_test, Y_pred) * 100

    ####### Genetic Algorithm
    ann = mlr.NeuralNetwork(hidden_nodes=[4, 3], activation='relu',
                        algorithm='genetic_alg',
                        pop_size=200, mutation_prob=0.1,
                        max_iters=max_iter, max_attempts=max_attempt,
                        bias=True, learning_rate=lr,
                        early_stopping=True, random_state=RANDOM_STATE, curve=True)

    ann.fit(X_train, Y_train)
    train_loss_ga = log_loss(Y_train, ann.predict(X_train))
    train_losses_ga.append(train_loss_ga)
    #Y_pred = ann.predict(X_test)
    #accuracy_ann = accuracy_score(Y_test, Y_pred) * 100

plt.figure()
plt.plot(max_iter_lst, train_losses_gd, marker='o', label='Gradient Descent')
plt.plot(max_iter_lst, train_losses_rhc, marker='o', label='RHC (8-Restart)')
plt.plot(max_iter_lst, train_losses_sa, marker='o', label='SA')
plt.plot(max_iter_lst, train_losses_ga, marker='o', label='GA')
plt.title('Training Loss v/s Iteration# (max_iter)')
plt.xlabel('Iteration#')
plt.ylabel("Training Loss (Log-Loss on Training Set")
plt.legend()
plt.grid()
plt.savefig('ann_train_loss.png')
plt.clf()