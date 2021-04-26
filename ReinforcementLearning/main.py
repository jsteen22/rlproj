import mdptoolbox.example, mdptoolbox.util
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import gym
import mlrose_hiive
from gym.envs.toy_text import frozen_lake
import random

def iters_con():





    return True

'''
P[s|a] = P[s’], s’, r
T[P[a]][P[s]][P[s][a][] =
'''


def gen_lake_env(lakesize,smallneg):
        #number of s
    #chance of burning?
    #lakesize = 4
    print('IN FROZEN')
    mapit = frozen_lake.generate_random_map(lakesize,p=0.9)
    #print(mapit)
    plt.figure()
    mapitnew = np.zeros((lakesize,lakesize))
    for index,row in enumerate(mapit):
        print(row)
        for index2,v in enumerate(row):
            if v == 'S':
                mapitnew[index][index2] = 0
            elif v == 'F':
                mapitnew[index][index2] = 1
            elif v == 'H':
                mapitnew[index][index2] = 2
            elif v == 'G':
                mapitnew[index][index2] = 3
    print(mapitnew)
    plt.matshow(mapitnew)
    plt.savefig('./lake' + str(lakesize) + '.png')
    plt.close()
    print('after map gen')
    env = frozen_lake.FrozenLakeEnv(desc=mapit,is_slippery=True)
    #print(env.nS)
    #print(env.desc)


    #env = gym.make('FrozenLake8x8-v0')
    print(np.reshape(env.desc,(lakesize,lakesize)))
    #print(sdsda)
    T = np.zeros((env.nA, env.nS, env.nS))
    R = np.zeros((env.nA, env.nS, env.nS))
    #print(env.P[0][0])
    #print()
    #print(env.nA)
    #print(env.nS)
    #print(env.env.P)
    for a in range(env.nA):
        for s in range(env.nS):
            for sp in range(len(env.P[s][a])):
                #sprint(env.P[s][a][sp][1])
                sprime = env.P[s][a][sp][1]
                T[a][s][sprime] = env.P[s][a][sp][0] + T[a][s][sprime]

                if smallneg == True and env.P[s][a][sp][3] == True and env.P[s][a][sp][2] == 0.0:
                    R[a][s][sprime] = -.1
                elif smallneg == True and env.P[s][a][sp][3] == False and env.P[s][a][sp][2] == 0.0:
                    R[a][s][sprime] = -.1
                elif smallneg == True:
                    R[a][s][sprime] = env.P[s][a][sp][2] + lakesize*100
                else:
                    R[a][s][sprime] = env.P[s][a][sp][2]
    #print(R[0])
    mdptoolbox.util.check(T,R)
    return env,T,R
def test_frozen_lake(policy,env):
    success = 0
    for i_eps in range(100):
        rewards = 0
        c = env.reset()
        for t in range(10000):
            c,reward,done,info = env.step(policy[c])
            if done:
                if reward == 1:
                    success = success + 1
                break
    return success
def test_forest(policy,forestsize,P,R):
    success = 0
    np.random.seed(42)
    states = np.random.randint(0,forestsize,size=100)
    #print(states)
    rewards = 0
    episrewards = 0
    maxt = 0
    for i_eps in states:
        ep_rewards = 0
        #s = np.random.randint(0,forestsize)
        burns = np.random.random(size=forestsize*10)
        s = i_eps
        for t in range(forestsize*10):
            #print(t)
            #print(s)
            a = policy[s]
            #print(a)
            rewards += R[s,a]
            #print(burns[t])
            if burns[t] < 1/(forestsize**.60) or a == 1:
                break
            elif s != forestsize-1:
                s = s + 1

            #if a
            '''
            if done:
                if reward == 1:
                    success = success + 1
                break
            '''
        s = i_eps

        for o in range(100):
            #print(s)
            a = policy[s]
            #print(a)
            episrewards += R[s,a]
            if burns[o] < 1/(forestsize**.60) or a == 1:
                s = 0
            elif s != forestsize-1:
                s = s + 1
    return rewards,episrewards

def prob_lake_size(lakesize,smallneg):

    #print(gabelegook)
    forestry = pd.DataFrame(columns=['Learner', 'Size', 'BurnProb', 'Result'])

    pops = [10,100,250,500,1000]
    disc = [.1, .25, .5, .75, .9 ,.99]
    disc = [.1, .5, .85, .99]
    vsuccess = []
    psuccess = []
    qsuccess = []
    viiters = []
    piiters = []
    qiters = []
    vdiffs = []
    pdiffs = []
    qdiffs = []
    vtime = []
    ptime = []
    qtime = []
    env,P,R = gen_lake_env(lakesize,smallneg)
    ppolicy = None
    qpolicy = None
    vpolicy = None
    psb = -1
    vsb = -1
    qsb = -1


    vi = mdptoolbox.mdp.ValueIteration(P,R,discount=1)
    pi = mdptoolbox.mdp.PolicyIteration(P,R,discount=.999,eval_type=0)
        #print('iter' + str(pi.iter))
        #q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=0000)
    vi.run()
    pi.run()
        #q.run()
        #print('iter' + str(pi.iter))
    viiters.append(vi.iter)
    piiters.append(pi.iter)
    vdiffs.append(vi.diffs)
    pdiffs.append(pi.diffs)
    psr = test_frozen_lake(pi.policy,env)
    if psr > psb:
        psb = psr
        ppolicy = pi.policy
    psuccess.append(psr)
    vsr = test_frozen_lake(vi.policy,env)
    if vsr > vsb:
        vsb = vsr
        vpolicy = vi.policy
    vsuccess.append(vsr)

    for d in disc:
        print(d)
        #print(1/p)
        #P,R = mdptoolbox.example.forest(S=p,r1=1000,p=(1/(p*500)))

        #R

        #print(np.reshape(pi.policy,(lakesize,lakesize)))

        #q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=0000)

        qavg = []
        qitera = []
        qdiffa = []
        qruns = 2
        sumarr = np.zeros(49999,)
        for i in range(qruns):
            q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=50000, explore='log',e=.33,schedule=None)
            q.run()
            qsr = test_frozen_lake(q.policy,env)
            if qsr > qsb:
                qsb = qsr
                qpolicy = q.policy
            qavg.append(qsr)

            qdiffa.append(q.diffs)
            sumarr = np.array(sumarr) + np.array(q.diffs)
            #qitera.append()


        qdiffs.append(sumarr/qruns)

        qsuccess.append(np.mean(qavg))
        qiters.append(22000)
        #qiter.append(np.mean(qitera))
        #print((vi.policy = 0).sum())

        #print(pi.policy)
        #print(q.policy)
        #print(vi.time)
        #print(pi.time)
        #print(q.time)
        #print(vi.V)
        #print(pi.V)
        #print(q.V)



    #plot the change in gamma to the success per the lake size
    '''
    plt.figure()
    plt.plot( disc, vsuccess, label='vi')
    plt.plot( disc, psuccess, label='pi')
    plt.plot( disc, qsuccess, label='q')
    plt.legend(loc='best')
    plt.savefig('./frozenlake/rewards/' + str(smallneg) +  str(lakesize)  +'.JPEG')
    '''
    #plot the differences by iteration for the most successful discount rate
    vin = 0
    pin = 0
    qin = 0
    for i in range(len(qsuccess)):
        if qsuccess[i] > qsuccess[qin]:
            qin = i

    print('3')
    plt.figure()
    plt.plot(range(len(qdiffs[qin])),qdiffs[qin],label = 'qi' + str(disc[qin]))
    plt.legend(loc='best')
    plt.savefig('./frozenlake/conv/q' + str(smallneg) +  str(lakesize)  +'.JPEG')
    plt.close()
    print('4')
    print('6')
    plt.figure()
    plt.plot( disc, qiters, label='q ' + str(disc[qin]))
    plt.legend(loc='best')
    plt.savefig('./frozenlake/iterdisc/q' + str(smallneg) +  str(lakesize)  +'.JPEG')
    plt.close()
    lc = ListedColormap(['m', 'k', 'c','r'])
    plt.matshow(np.reshape(qpolicy,(lakesize,lakesize)),cmap=lc)
    plt.savefig('./qoptimal' + str(lakesize) +'.png')
    plt.close()

    #print(np.reshape(qpolicy,(lakesize,lakesize)))
    print(q.Q)
    print("q %6f %.9f %3f %.2f" % (50000, q.time, qsuccess[qin],disc[qin]))

    return True


def run_pv_lake(lakesize,smallneg):
    #pops = [10,100,250,500,1000]
    #disc = [.1, .25, .5, .75, .9 ,.99]
    #disc = [.1, .5, .85, .99]
    vsuccess = []
    psuccess = []
    qsuccess = []
    viiters = []
    piiters = []
    qiters = []
    vdiffs = []
    pdiffs = []
    qdiffs = []
    vtime = []
    ptime = []
    qtime = []
    env,P,R = gen_lake_env(lakesize,smallneg)
    ppolicy = None
    qpolicy = None
    vpolicy = None
    psb = -1
    vsb = -1
    qsb = -1
    if lakesize == 22:
        eps = .001
    else:
        eps = .01

    vi = mdptoolbox.mdp.ValueIteration(P,R,discount=1,epsilon=eps)
    pi = mdptoolbox.mdp.PolicyIteration(P,R,discount=.999)
        #print('iter' + str(pi.iter))
        #q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=0000)
    vi.run()
    pi.run()
        #q.run()
        #print('iter' + str(pi.iter))
    vi.iter
    pi.iter
    vi.diffs
    pi.diffs
    psr = test_frozen_lake(pi.policy,env)
    ppolicy = pi.policy
    psuccess.append(psr)
    vsr = test_frozen_lake(vi.policy,env)
    vpolicy = vi.policy

    plt.figure()
    plt.plot(range(1,len(vi.diffs)+1),vi.diffs,label = 'vi')
    plt.legend(loc='best')
    plt.savefig('./frozenlake/conv/v' + str(smallneg) +  str(lakesize)  +'.JPEG')
    plt.close()
    print('5')
    plt.figure()
    plt.plot(range(101,len(pi.diffs)+1),pi.diffs[100:],label = 'pi' )
    plt.legend(loc='best')
    plt.savefig('./frozenlake/conv/1p' + str(smallneg) +  str(lakesize)  +'.JPEG')
    plt.close()
    print("vi %6f %.9f %3f" % (vi.iter, vi.time, vsr))
    print("pi %6f %.9f %3f" % (pi.iter, pi.time, psr))
    print(np.reshape(pi.policy,(lakesize,lakesize)))
    print(np.reshape(vi.policy,(lakesize,lakesize)))
    np.asarray(vi.policy)
    print((np.asarray(vi.policy) != np.asarray(pi.policy)).sum())
    lc = ListedColormap(['m', 'k', 'c','r'])
    plt.matshow((np.reshape(vi.policy,(lakesize,lakesize))),cmap=lc)
    plt.savefig('./vioptimal' + str(lakesize) +'.png')
    plt.close()
    lc = ListedColormap(['m', 'k', 'c','r'])
    plt.matshow(np.reshape(pi.policy,(lakesize,lakesize)),cmap=lc)
    plt.savefig('./pioptimal' + str(lakesize) +'.png')
    plt.close()

def run_pv_forest(forestsize,P,R):
    #pops = [10,100,250,500,1000]
    #disc = [.1, .25, .5, .75, .9 ,.99]
    #disc = [.1, .5, .85, .99]
    vsuccess = []
    psuccess = []
    qsuccess = []
    viiters = []
    piiters = []
    qiters = []
    vdiffs = []
    pdiffs = []
    qdiffs = []
    vtime = []
    ptime = []
    qtime = []
    ppolicy = None
    qpolicy = None
    vpolicy = None
    psb = -1
    vsb = -1

    disc = [.1, .25, .5, .75 ,.99]
    vsuccesst = []
    psuccesst = []
    vsuccess1 = []
    psuccess1 = []
    viiter1 = []
    piiter = []
    qiter = []
    for d in disc:
        vi = mdptoolbox.mdp.ValueIteration(P,R,discount=d)
        pi = mdptoolbox.mdp.PolicyIteration(P,R,discount=d)
        #q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=50000)
        vi.run()
        pi.run()
        psr = test_forest(pi.policy,forestsize,P,R)
        #ppolicy = pi.policy
        vsr = test_forest(vi.policy,forestsize,P,R)
        vsuccesst.append(vsr[0])
        psuccesst.append(psr[0])
        vsuccess1.append(vsr[1])
        psuccess1.append(psr[1])
        if d > .89:
            print(vsr)
            print(psr)
        #q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=0000)
        '''
        qavg = []
        qruns = 2
        for i in range(qruns):
            q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=22000, explore='epsilon',e=.35)
            q.run()
            qavg.append(test_forest(q.policy,env))
        qsuccess.append(np.mean(qavg))
        '''
        #print((vi.policy = 0).sum())

        #print(pi.policy)
        #print(q.policy)
        #print(vi.time)
        #print(pi.time)
        #print(q.time)
        #print(vi.V)
        #print(pi.V)
        #print(q.V)

    print(pi.policy)
    print(vi.policy)
    plt.figure()
    plt.plot( disc, vsuccesst, label='vi')
    plt.plot( disc, psuccesst, label='pi')
    plt.plot(disc, vsuccess1, label = 'vi 100')
    plt.plot( disc, psuccess1, label = 'pi 100')
    #plt.plot( disc, qsuccess, label='q')
    plt.legend(loc='best')
    plt.savefig('./forest/disc' +  str(forestsize)  +'.JPEG')
    plt.close()
    print("vi %6f %.9f %9f %9f" % (vi.iter, vi.time, vsr[0],vsr[1]))
    print("pi %6f %.9f %9f %9f" % (pi.iter, pi.time, psr[0],psr[1]))

    plt.figure()
    plt.plot(range(1,len(vi.diffs)+1),vi.diffs,label = 'vi')
    plt.legend(loc='best')
    plt.savefig('./forest/conv/v' +  str(forestsize)  +'.JPEG')
    plt.close()
    plt.figure()
    plt.plot(range(1,len(pi.diffs)+1),pi.diffs,label = 'pi')
    plt.legend(loc='best')
    plt.savefig('./forest/conv/p' +  str(forestsize)  +'.JPEG')
    plt.close()
    #plt.figure(figsize=(100,200))
    plt.matshow((np.reshape(vi.policy,(forestsize,1))),aspect='auto')
    plt.savefig('./vforest' + str(forestsize) +'.png')
    plt.close()
    plt.matshow((np.reshape(pi.policy,(forestsize,1))),aspect='auto')
    plt.savefig('./pforest' + str(forestsize) +'.png')
    plt.close()
    '''

    psuccess.append(psr)
    plt.figure()
    plt.plot(range(1,len(vi.diffs)+1),vi.diffs,label = 'vi')
    plt.legend(loc='best')
    plt.savefig('./forest/conv/v' + str(smallneg) +  str(lakesize)  +'.JPEG')
    plt.close()
    print('5')
    plt.figure()
    plt.plot(range(101,len(pi.diffs)+1),pi.diffs[100:],label = 'pi' )
    plt.legend(loc='best')
    plt.savefig('./forest/conv/1p' + str(smallneg) +  str(lakesize)  +'.JPEG')
    plt.close()
    print("vi %6f %.9f %3f" % (vi.iter, vi.time, vsr))
    print("pi %6f %.9f %3f" % (pi.iter, pi.time, psr))
    print(np.reshape(pi.policy,(lakesize,lakesize)))
    print(np.reshape(vi.policy,(lakesize,lakesize)))
    np.asarray(vi.policy)
    print((np.asarray(vi.policy) != np.asarray(pi.policy)).sum())
    lc = ListedColormap(['m', 'k', 'c','r'])
    plt.matshow((np.reshape(vi.policy,(lakesize,lakesize))),cmap=lc)
    plt.savefig('./vioptimal' + str(lakesize) +'.png')
    plt.close()
    lc = ListedColormap(['m', 'k', 'c','r'])
    plt.matshow(np.reshape(pi.policy,(lakesize,lakesize)),cmap=lc)
    plt.savefig('./pioptimal' + str(lakesize) +'.png')
    '''
    plt.close()
def prob_forest_size(forestsize,P,R):
    disc = [.1, .25, .5, .75, .9 ,.99]
    disc = [.1]
    vsuccess = []
    psuccess = []
    qsuccesst = []
    qsuccess1 = []
    viiters = []
    piiters = []
    qiters = []
    vdiffs = []
    pdiffs = []
    qdiffs = []
    vtime = []
    ptime = []
    qtime = []
    ppolicy = None
    qpolicy = None
    vpolicy = None
    psb = -1
    vsb = -1
    qsb = -1

    '''
    vi = mdptoolbox.mdp.ValueIteration(P,R,discount=1)
    pi = mdptoolbox.mdp.PolicyIteration(P,R,discount=.999,eval_type=0)
        #print('iter' + str(pi.iter))
        #q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=0000)
    vi.run()
    pi.run()
        #q.run()
        #print('iter' + str(pi.iter))
    viiters.append(vi.iter)
    piiters.append(pi.iter)
    vdiffs.append(vi.diffs)
    pdiffs.append(pi.diffs)
    psr = test_frozen_lake(pi.policy,env)
    if psr > psb:
        psb = psr
        ppolicy = pi.policy
    psuccess.append(psr)
    vsr = test_frozen_lake(vi.policy,env)
    if vsr > vsb:
        vsb = vsr
        vpolicy = vi.policy
    vsuccess.append(vsr)
    '''

    for d in disc:
        print(d)
        #print(1/p)
        #P,R = mdptoolbox.example.forest(S=p,r1=1000,p=(1/(p*500)))

        #R

        #print(np.reshape(pi.policy,(lakesize,lakesize)))

        #q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=0000)

        qavgt = []
        qavg1 = []
        qitera = []
        qdiffa = []
        qruns = 3
        sumarr = np.zeros(49999,)
        for i in range(qruns):
            q = mdptoolbox.mdp.QLearning(P,R,discount=.1,n_iter=50000, explore='log',e=.33,schedule=None)
            q.run()
            qsr = test_forest(q.policy,forestsize, P,R)
            if qsr[0] > qsb:
                qsb = qsr[0]
                qpolicy = q.policy
            qavgt.append(qsr[0])
            qavg1.append(qsr[1])
            qdiffa.append(q.diffs)
            sumarr = np.array(sumarr) + np.array(q.diffs)
            #qitera.append()


        qdiffs.append(sumarr/qruns)

        qsuccesst.append(np.mean(qavgt))
        qsuccess1.append(np.mean(qavg1))
        qiters.append(50000)
    #plot the differences by iteration for the most successful discount rate
    vin = 0
    pin = 0
    qin = 0
    '''
    for i in range(len(qsuccess)):
        if qsuccess[i] > qsuccess[qin]:
            qin = i
    '''
    print('3')
    plt.figure()
    plt.plot(range(len(qdiffs[qin])),qdiffs[qin],label = 'qi' + str(disc[qin]))
    plt.legend(loc='best')
    plt.savefig('./forest/conv/q' +  str(forestsize)  +'.JPEG')
    plt.close()
    print('4')
    print('6')
    plt.figure()
    plt.plot( disc, qiters, label='q ' + str(disc[qin]))
    plt.legend(loc='best')
    plt.savefig('./forest/iterdisc/q' +  str(forestsize)  +'.JPEG')
    plt.close()
    plt.matshow(np.reshape(qpolicy,(forestsize,1)),aspect='auto')
    plt.savefig('./qforest' + str(forestsize) +'.png')
    plt.close()

    #print(np.reshape(qpolicy,(lakesize,lakesize)))
    print(q.Q)
    print("q %6f %.9f %8f %8f %.2f" % (50000, q.time, qsuccesst[qin],qsuccess1[qin],disc[qin]))
    return True



def q_learning_tuning(lakesize,smallneg,constant,env,P,R):
    #env = frozen_lake.generate_random_map(lakesize)
    #print(env)
    exploreStrats = [(None,'log'),(mlrose_hiive.ExpDecay(),'exp'), (mlrose_hiive.GeomDecay(),'geom')]#, (None,'log')]
    #schedule.evaluate(iters)
    decays = [.1, .33, .5, .66, .85, .99]
    decays = [.1, .5, .85, .99]
    if constant == True:
        exploreStrats = [.15, .25, .33, .5]#, .66, .85]
        lr = [.5, .75, .85, .95]
        print('constant epsilon')
    for es in exploreStrats:
    #env = frozen_lake.FrozenLakeEnv(desc=env,is_slippery=True)
        qruns = 3
        for d in decays:
            qavg = []
            for i in range(qruns):
                if constant == True:
                    q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=50000, explore='epsilon',e=es,schedule=None)
                else:
                    q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=50000, explore='log',e=.65,schedule=es[0])
                q.run()
                qavg.append(test_frozen_lake(q.policy,env))
            if constant == False:
                print(es[1] + ' ' + str(lakesize) + ' ' + str(d) + ' ' + str(np.mean(qavg) ) )
            else:
                print(str(es) + ' ' + str(lakesize) + ' ' + str(d) + ' ' + str(np.mean(qavg)))
    env.close()
    return True

def forest_q_learning_tuning(forestsize,smallneg,constant,P,R):
    #env = frozen_lake.generate_random_map(lakesize)
    #print(env)
    exploreStrats = [(None,'log'),(mlrose_hiive.ExpDecay(),'exp'), (mlrose_hiive.GeomDecay(),'geom')]#, (None,'log')]
    #schedule.evaluate(iters)
    decays = [.1, .33, .5, .66, .85, .99]
    decays = [.1, .5, .85, .99]
    if constant == True:
        exploreStrats = [.15, .25, .33, .5]#, .66, .85]
        lr = [.5, .75, .85, .95]
        print('constant epsilon')
    for es in exploreStrats:
    #env = frozen_lake.FrozenLakeEnv(desc=env,is_slippery=True)
        qruns = 3
        for d in decays:
            qavgt = []
            qavg1 = []
            for i in range(qruns):
                if constant == True:
                    q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=50000, explore='epsilon',e=es,schedule=None)
                else:
                    q = mdptoolbox.mdp.QLearning(P,R,discount=d,n_iter=50000, explore='log',e=.65,schedule=es[0])
                q.run()
                qres = test_forest(q.policy,forestsize,P,R)
                qavgt.append(qres[0])
                qavg1.append(qres[1])
            if constant == False:
                print(es[1] + ' ' + str(forestsize) + ' ' + str(d) + ' ' + str(np.mean(qavgt)) + ' ' + str(np.mean(qavg1)) )
            else:
                print(str(es) + ' ' + str(forestsize) + ' ' + str(d) + ' ' + str(np.mean(qavgt)) + ' ' +  str(np.mean(qavg1)))
    return True
'''
lakes = [ 4]
for ls in lakes:
    print('loop start')
    env,P,R = gen_lake_env(ls,False)
    #prob_lake_size(ls,False)
    prob_lake_size(ls,False)
    #q_learning_tuning(ls,False,True,env,P,R)
    #q_learning_tuning(ls,False,False,env,P,R)
    #env.render()
    #run_pv_lake(ls,False)
    env.close()

    #print(ls)
'''
forests = [64,666]

for f in forests:
    P,R = mdptoolbox.example.forest(S=f,r1=f**1.2,r2=1,p=1/(f**.60))
    #run_pv_forest(f,P,R)
    #forest_q_learning_tuning(f,False,True,P,R)
    #forest_q_learning_tuning(f,False,False,P,R)
    prob_forest_size(f,P,R)