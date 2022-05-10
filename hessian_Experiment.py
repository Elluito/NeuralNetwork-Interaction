import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as l
import tensorflow_addons as tfa
import  tensorflow.keras as keras
from collections import namedtuple
import sympy as sym
import gym
import re

import os
import time
import multiprocessing
from multiprocessing import Process

BATCH_SIZE = 8
RECORD = False
PATH_TO_FOLDER = "SGD_experiments/"
import functools

def build_model(input_shape,output_dim,type="actor",mode="sup"):
    layers = []
    if type == "actor":
        layers = [l.InputLayer(input_shape,dtype=tf.int32),
                  l.Flatten(),
            l.Dense(128,activation=gelu),
            l.Dense(128, activation=gelu),
            l.Dense(128, activation=gelu),
            l.Dense(output_dim,activation="softmax")
            ]
    if type == "critic":
        layers = [l.InputLayer(input_shape, dtype=tf.float32),
                  l.Flatten(),
                  l.Dense(128, activation="relu"),
                  l.Dense(output_dim)
                  ]

    if type == "fc":
        layers = [l.InputLayer(input_shape,dtype=tf.float32),
                  l.Reshape((91,)),
                l.Dense(512,activation="tanh",input_shape=(91,)),
                l.BatchNormalization(),
                l.Dropout(0.5),
                l.Dense(512, activation="tanh"),
                l.BatchNormalization(),
                l.Dropout(0.5),
                l.Dense(output_dim,activation="softmax")

        ]

    model = tf.keras.Sequential(layers)
    if mode == "sup":
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

    else:
        model.build(input_shape=input_shape)

    return model

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        index = range(len(self.memory))
        real_index = np.random.choice(index, batch_size)
        return_list=[]
        for i in real_index:
            return_list.append(self.memory[i])
        return return_list

    def __len__(self):
        return len(self.memory)

def relu(x):
    return sym.Piecewise((0,x<0),(x,x>=0))
def hubert_loss(x):
    return sym.Piecewise((x**2,x<1),(sym.Abs(x),x>=1))
def MSE_loss(x):
    return (x)**2
def recursive_finding(expresion,current_elements):
    hijos = expresion.args
    for hijo in hijos:
        if hasattr(hijo,"indices"):
            current_elements.append(hijo)
        if len(hijo.args) != 0:
            recursive_finding(hijo,current_elements)

def find_variables(symbolic_expresion):
    # hijos = symbolic_expresion.args
    expressions =[]
    # temp=[]
    #
    # pool = multiprocessing.Pool(4)
    # zip(*pool.map(lambda x: recursive_finding(x,temp), hijos))
    recursive_finding(symbolic_expresion,expressions)

    return expressions
def single_step(current_expression,symbols_to_replace,x_batch,y_batch,average_kernel,index):
    for symbol in symbols_to_replace:
        name = symbol.free_symbols.pop().name
        if "kernel" in name:
            thing = symbol.free_symbols.pop()
            indices = symbol.indices
            current_expression = current_expression.subs(thing[indices], average_kernel[name][indices])
        if name == "y":
            thing = symbol.free_symbols.pop()
            current_expression = current_expression.subs(thing[0], y_batch[index][0])
            current_expression = current_expression.subs(thing[1], y_batch[index][1])
        if name == "X":
            thing = symbol.free_symbols.pop()
            current_expression = current_expression.subs(thing[0], x_batch[index][0])
            current_expression = current_expression.subs(thing[1], x_batch[index][1])
            current_expression = current_expression.subs(thing[2], x_batch[index][2])
            current_expression = current_expression.subs(thing[3], x_batch[index][3])
    return current_expression.evalf()

"""
This function sums for all the values of x and y in the batch. it needs the average of the weights of the network 
that we already computed.
"""


def splitlist(inlist, chunksize):
    return [inlist[x:x + int(chunksize)] for x in range(0, len(inlist), int(chunksize))]

def process_batch(symbolic_expression,symbols_to_replace,x_batch,y_batch,average_kernel,queue):
    total_sum = 0

    for index in range(len(x_batch)):
        thing = single_step(symbolic_expression,symbols_to_replace,x_batch,y_batch,average_kernel,index)
        total_sum += thing
    queue.put(float(total_sum))
def obtain_value_from_batch(symbolic_expression,x_batch,y_batch,average_kernel):

    total_sum = 0
    current_expression = symbolic_expression
    x_batch = np.squeeze(x_batch)
    y_batch = np.squeeze(y_batch)
    t_0 = time.time()
    symbols_to_replace = find_variables(symbolic_expression)
    t_1 = time.time()
    print("find_symbols time: {}".format(t_1-t_0))
    # q = multiprocessing.Queue()
    # processes = []
    # rets = []
    # sub_batches_x = splitlist(x_batch,len(x_batch)/os.cpu_count())
    # sub_batches_y = splitlist(y_batch, len(y_batch) / os.cpu_count())
    #
    #
    # for i in range(len(sub_batches_x)):
    #     sub_x = sub_batches_x[i]
    #     sub_y = sub_batches_y[i]
    #     p = Process(target=process_batch, args=(current_expression,symbols_to_replace,sub_x,sub_y,average_kernel,q))
    #     p.Daemon = True
    #     processes.append(p)
    #     p.start()
    # for p in processes:
    #     ret = q.get()  # will block
    #     rets.append(ret)
    # for p in processes:
    #     p.join()
    # return sum(rets)
    values = {}
    add_x = False
    add_y = False
    x_symbols = []
    y_symbols = []
    for elem in symbols_to_replace:
        name = elem.free_symbols.pop().name
        if "kernel" in name:
            thing = elem.free_symbols.pop()
            indices = elem.indices
            values[thing[indices]]= average_kernel[name][indices]
        if name == "y":
            thing = elem.free_symbols.pop()

            values[thing[0]] = 0
            values[thing[1]] = 0
            if not add_y:
                y_symbols.append(thing[0])
                y_symbols.append(thing[1])
                add_y=True
        if name == "X":
            thing = elem.free_symbols.pop()
            if not add_x:
                x_symbols.append(thing[0,0])
                x_symbols.append(thing[1,0])
                x_symbols.append(thing[2,0])
                x_symbols.append(thing[3,0])
                add_x = True
            values[thing[0]] = 0
            values[thing[1]] = 0
            values[thing[2]] = 0
            values[thing[3]] = 0



    t0 = time .time()
    for index in range(len(x_batch)):

        for i,val in enumerate(y_symbols):
            values[val] = y_batch[index][i]
        for i,val in enumerate(x_symbols):
            values[val]= x_batch[index][i]
        current_expression = current_expression.subs(values)
        # for symbol in symbols_to_replace:
        #     name = symbol.free_symbols.pop().name
        #     if "kernel" in name:
        #         thing = symbol.free_symbols.pop()
        #         indices = symbol.indices
        #         current_expression = current_expression.subs(thing[indices],average_kernel[name][indices])
        #     if name == "y":
        #         thing = symbol.free_symbols.pop()
        #         current_expression = current_expression.subs(thing[0],y_batch[index][0])
        #         current_expression = current_expression.subs(thing[1],y_batch[index][1])
        #     if name == "X":
        #         thing = symbol.free_symbols.pop()
        #         current_expression = current_expression.subs(thing[0], x_batch[index][0])
        #         current_expression = current_expression.subs(thing[1], x_batch[index][1])
        #         current_expression = current_expression.subs(thing[2], x_batch[index][2])
        #         current_expression = current_expression.subs(thing[3], x_batch[index][3])
        total_sum += current_expression.evalf()
    t1 = time.time()
    print("Time for loop replacing expresion: {}".format(t1-t0))
    return total_sum
def divide_matrix(max_index,number_to_divide):
    import math


    assert math.log(number_to_divide, 2).is_integer(), "The number must be a power of 2"
    n = int(max_index)/int(number_to_divide)
    matrix = []
    for i in range(max_index):
        for j in range(max_index):
            matrix.append((i,j))


    out = splitlist(matrix,n)
    return out
def run_section(variables,seccion,f,x_batch,y_batch,average_kernel,dictionary):

    for element in seccion:
        i,j = element
        df_dij = f.diff(variables[i]).diff(variables[j])
        t_0 = time.time()
        number = obtain_value_from_batch(df_dij, x_batch, y_batch, average_kernel=average_kernel)
        t_1 = time.time()
        print("Time for element {},{} :{}".format(i, j, t_1 - t_0))
        dictionary[i,j] = float(number)




"""
FUNCTION that returns the hessian matrix as a function to be evaluated in certain point 
"""
def hessian_function(model,loss_function="MSE",activations=[],x_batch=[],y_batch=[]):
    input_shape = model.input_shape
    X = sym.MatrixSymbol("X",input_shape[1],input_shape[2])
    X = sym.Matrix(X)
    X = X.row_insert(0, sym.Matrix([1]))



    variables = model.variables
    groups = {}
    for var in variables:
        name = var.name
        if groups.keys():
            number = re.findall(r'\d+', name)[0]
            if number in groups.keys():
                groups[number].append(var)
            else:
                groups[number] = [var]


        else:
            number = re.findall(r'\d+',name)
            groups[number[0]] = [var]
    variables = []
    average_of_weights = {}
    step = 0
    temp = None
    for key in groups.keys():
        group = groups[key]

        for var in group:

           if "kernel" in var.name:

                shape = list(var.shape)
                shape.reverse()
                ## AGREGGATE TYHE BIAS TO DE COLUM OF THE KERNELS SINCE W_(n,m)*X_(m,1)=Z_(n,1)
                shape[1]+=1
                name = "kernel_{}".format(key)
                cosa = [10]
                cosa.extend(shape)
                points = np.random.uniform(-np.sqrt(6) / np.sqrt(shape[0] + shape[1]),
                                          np.sqrt(6) / np.sqrt(shape[0] + shape[1]),
                                          cosa)
                average_of_weights[name] = np.mean(points,axis=0)
                W_i = sym.MatrixSymbol(name, *shape)
                W_i = sym.Matrix(W_i)
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        variables.append(W_i[i,j])
                if step == 0:
                    if activations[step] == "relu":
                        try:
                            l = W_i*X
                            temp = l.applyfunc(relu)

                        except:
                            l = W_i * X.T
                            temp = l.applyfunc(relu)


                    if activations[step] == "linear":
                        try:
                            temp = W_i*X


                        except:
                            temp = W_i * X.T


                else:
                    temp = temp.row_insert(0, sym.Matrix([1]))
                    if activations[step] == "relu":

                        try:

                            l = W_i * temp
                            l.applyfunc(relu)

                        except:
                            l = W_i * temp.T
                            l.applyfunc(relu)


                    if activations[step] == "linear":
                        try:
                            temp = W_i * temp


                        except:
                            temp = W_i * temp.T

        step += 1
        ###Now we have the output of the neural network thus we only need the loss function
    if loss_function == "MSE":
        shape = list(model.output_shape)
        shape.insert(0, 1)
        shape.remove(None)

        y = sym.MatrixSymbol("y", *shape)
        y = sym.Matrix(y)
        k = None
        try:
            k = y-temp
        except:
            k = y.T-temp

        f = k.applyfunc(MSE_loss)

        # f = f[0]







        matrix_function = []
        i = 0
        j = 0
        sub_matrices = divide_matrix(len(variables),os.cpu_count())
        matrix = {}
        processes = []
        for i in range(len(sub_matrices)):
            p = Process(target=run_section, args=(variables,sub_matrices[i],f,x_batch,y_batch,average_of_weights,matrix))
            p.Daemon = True
            processes.append(p)
            p.start()
        for p in processes:
            p.join()


        # for var_i in variables:
        #     matrix_function.append([])
        #     for var_j in variables:
        #
        #         df_dij = f.diff(var_i).diff(var_j)
        #         t_0 = time.time()
        #         number = obtain_value_from_batch(df_dij,x_batch,y_batch,average_kernel=average_of_weights)
        #         t_1 = time.time()
        #         print("Time for element {},{} :{}".format(i,j,t_1-t_0))
        #         matrix_function[i].append(float(number))
        #         j+=1

            # i += 1
        return matrix
    if loss_function == "huber":
        shape = list(model.output_shape)
        shape.insert(0,1)
        shape.remove(None)

        y = sym.MatrixSymbol("y",*shape)
        y = sym.Matrix(y)
        k=None
        try:
            k = y-temp
        except:
            k = y.T-temp

        # f = k.applyfunc(hubert_loss)

        f = hubert_loss(k[0])+hubert_loss(k[1])
        print(f)
        return f




class Policy:
    # __slots__ = ( 'width', 'height', 'dim_action', 'gamma','load_name','use_prior','use_image','model','memory','epsilon','escala','mapeo','state_space','priority','priority_memory','action_space')

    def __init__(self,input_dim=(91,1,),output_dim=7*7, gamma=0.99, load_name=None,use_prior =False):
        # tf.enable_eager_execution()


        # tf.logging.set_verbosity(tf.logging.ERROR)
        self.priority = use_prior
        self.gamma = gamma
        self.value_memory = ReplayMemory(100000)
        self.action_dim = output_dim
        self.input_space = input_dim
        self.epsilon_ini = 0.9
        self.epsilon_end = 0.1
        self.factor = np.exp((np.log(self.epsilon_end)-np.log(self.epsilon_end)/200))
        self.count = 1

        # self.model = build_model(input_dim,output_dim,mode="no_sup")
        self.model= build_model(input_dim,output_dim,type="critic",mode="no_sup")
        self.model.summary()
        # self.model.load_weights("../models/DOMINATOR_e31-val_loss_2.2780.hdf5")
        self.optimizer = tf.keras.optimizers.SGD(lr=0.1,momentum=0.5)
            # self.model.compile(loss=tf.compat.v1.losses.huber_loss, optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0002,momentum=0.01))


        if load_name is not None: self.model = keras.models.load_model(load_name)






        # Episode policy and reward history

    # @tf.function
    def func(self,y_true, y_pred):
        errors = tf.pow(tf.reduce_sum(y_true- y_pred, axis=1), 2)
        print(self.pesos)

        loss = tf.reduce_mean(tf.multiply(self.pesos, errors))
        return loss

    def load_Model(self, load_name=None):
        self.model.load_weights(load_name)


    def saveModel(self, name):

        self.model.save_weights(name + '.hdf5')

    def reset(self):
        self.count = 1
    def reset_model(self):
        self.model = build_model(self.input_space, self.action_dim, type="critic", mode="no_sup")

    def take_action(self,observation):
        p = self.model.predict(observation)[0]
        thing = self.epsilon_ini*self.factor**self.count
        e = thing if thing > self.epsilon_end else self.epsilon_end
        self.count += 1
        return np.argmax(p) if np.random.rand() > e else np.random.choice(range(len(p)))

    # @tf.function
    def update_policy(self,name,mode="SGD"):
        if not self.priority:


                transitions = self.value_memory.sample(BATCH_SIZE)
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.

                shape = [-1]
                shape.extend(self.input_space)
                batch = Transition(*zip(*transitions))
                state_batch = batch.state
                state_batch = np.array(state_batch, dtype=np.float64).reshape(shape)
                action_batch = np.array([list(range(len(batch.action))), list(batch.action)]).transpose()
                reward_batch = np.array(batch.reward)
                reward_batch = (reward_batch - np.mean(reward_batch)) / (np.std(reward_batch) + 0.001)

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = ~np.array(batch.next_terminal)
                non_final_mask = np.nonzero(non_final_mask)[0]
                non_final_next_states = np.array(batch.next_state)[non_final_mask]

                next_state_values = np.zeros([BATCH_SIZE], dtype=float)
                non_final_next_states = np.array(non_final_next_states, dtype=np.float64).reshape(shape)
                next_state_values[non_final_mask] = np.max(np.array(self.model.predict([non_final_next_states])),
                                                           axis=1)
                q_update = (reward_batch + self.gamma * next_state_values)
                q_values = np.array(self.model.predict_on_batch([state_batch]))
                q_values[action_batch[:, 0], action_batch[:, 1]] = q_update

                critic = self.model


                loss_object = tf.compat.v1.losses.mean_squared_error

                if mode == "SGD":

                    with  tf.GradientTape(persistent=True) as tape:



                        q_value_state = critic(state_batch.reshape(shape),training=True)



                        loss_critic = loss_object(q_values,q_value_state)



                    gradients2 = tape.gradient(loss_critic,critic.trainable_variables)
                    del tape
                    VALOR = loss_critic.numpy()
                    self.optimizer.apply_gradients(zip(gradients2, critic.trainable_variables))
                    if RECORD:
                        with open(name,"a") as f:
                            f.write("{}\n".format(str(VALOR)))                # print("Policy Loss: {}".format(


                if mode == "hessian":

                    cosa = hessian_function(critic,loss_function="MSE",activations = ["relu","linear"],
                                            x_batch=state_batch,y_batch=q_values)
                    with  tf.GradientTape(persistent=True) as tape:
                        q_value_state = critic(state_batch.reshape(shape), training=True)

                        loss_critic = loss_object(q_values, q_value_state)

                    gradients2 = tape.gradient(loss_critic, critic.trainable_variables)
                    del tape
                    VALOR = loss_critic.numpy()
                    self.optimizer.apply_gradients(zip(gradients2, critic.trainable_variables))
                    if RECORD:
                        with open(name, "a") as f:
                            f.write("{}\n".format(str(VALOR)))





        else:
            if len(self.priority_memory) < BATCH_SIZE:
                return
            obs_batch, act_batch, rew_batch, next_obs_batch, not_done_mask, weights, indxes = self.priority_memory.sample(BATCH_SIZE,0.5)
            self.pesos = np.array(weights,dtype=np.float32)
            non_final_mask = np.where(not_done_mask == 0)[0]
            act_batch = np.array([list(range(len(act_batch))), act_batch]).transpose()
            next_state_values = np.zeros([BATCH_SIZE], dtype=float)
            next_state_values[non_final_mask] = np.max(self.model.predict(next_obs_batch[non_final_mask]), axis=1)

            rew_batch = (rew_batch - np.mean(rew_batch)) / (np.std(rew_batch) + 0.001)
            # rew_batch = rew_batch/max(np.abs(rew_batch))

            q_update = (rew_batch + self.gamma * next_state_values)
            q_values = np.array(self.model.predict([obs_batch]))
            q_values[act_batch[:, 0], act_batch[:, 1]] = q_update

            with tf.GradientTape() as tape:
                # tape.watch(self.model.trainable_variables)
                y_pred = self.model([obs_batch],training=True)


                errors = tf.pow(tf.reduce_sum(q_values-y_pred,axis=1),2)

                loss = tf.reduce_mean(tf.multiply(weights,errors))
                loss = tf.reduce_mean(errors)

            grads = tape.gradient(loss, self.model.trainable_variables)
            del tape

            # grads = self.optimizer.compute_gradients(f,self.model.trainable_variables)


            # for i,elem in enumerate(grads):
            #     grads[i] =elem[1].numpy()

            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))



            # salidas = self.model.fit(obs_batch, q_values, batch_size=len(q_values), epochs=20, verbose=0)
            # print(salidas.history["loss"])
            td_error = self.model.predict([obs_batch])[act_batch[:, 0], act_batch[:, 1]]-q_update
            self.priority_memory.update_priorities(indxes, abs(td_error))
    def add_transition(self,*args):
        self.episodic_memory.append(Transition(*args))




import numpy as np
import gym




if __name__ == '__main__':
    import datetime as date
    env1 = "CartPole-v1"
    env2 = "MountainCar-v0"
    env3 = "Acrobot-v1"
    env4 = "Pendulum-v0"
    env5 = "LunarLander-v2"
    env_name = env1
    env = gym.make(env_name)
    Transition = namedtuple('Transition',
                            ('state', 'action', 'reward',"next_state","next_terminal"))
    t = date.datetime.now()
    agent = Policy((env.observation_space.shape[0],1,),env.action_space.n)
    # agent.load_Model("model_{}_episodes_{}.hdf5".format(env2,1000))
    # agent.load_Model("model_{}_episodes_{}.hdf5".format(env2,300))
    number_of_experiments = 1
    episodes = 1
    # actions_meaning = ["left","no move","right"]
    # agent.load_Model("model_{}_episodes_{}.hdf5".format(env1,2000))
    for exp in range(number_of_experiments):
        t= str(date.datetime.now()).replace(":","-")
        nombre = PATH_TO_FOLDER+"REWARD_experiment_{}_time{}.txt".format(exp,t)
        if RECORD:
            open(nombre,"w").close()
        for episode in range(episodes):
            observation = env.reset()
            transitions = []
            done = False
            episode_reward =[]
            while not done:

                action = agent.take_action(np.array(observation).reshape(1,env.observation_space.shape[0],1))# your agent here (this takes random actions)
                # print("Action taken: {}".format(actions_meaning[action]))
                next_observation, reward, done, info = env.step(action)
                episode_reward.append(reward)
                transitions.append(Transition(observation, action, reward, next_observation,done))
                agent.value_memory.push(observation, action, reward, next_observation,done)
                observation = next_observation
                if done:
                    observation = env.reset()
                agent.update_policy(PATH_TO_FOLDER+"LOSS_experiment_{}_time{}.txt".format(exp,t),mode="hessian")
            agent.reset()

            print("Episode: {}".format(episode))
            print("Sum reward: {} ".format(sum(episode_reward)))
            if RECORD:
                with open(nombre,"a") as f:
                    f.write("{}\n".format(sum(episode_reward)))
            if episode%10 == 0:
                agent.saveModel(PATH_TO_FOLDER+"MODEL_experiment_{}_date_{}".format(exp,t))
            agent.reset_model()



    # for episode in range(10):
    #     observation = env.reset()
    #     transitions = []
    #     done = False
    #     while not done:
    #         env.render()
    #         action = agent.take_action(np.array(observation).reshape(1,env.observation_space.shape[0],1))# your agent here (this takes random actions)
    #         next_observation, reward, done, info = env.step(action)
    #
    #         observation = next_observation
    #         if done:
    #             observation = env.reset()

    env.close()