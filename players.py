""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle as pickle
import cv2
import pong_game
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib import animation
from JSAnimation.IPython_display import display_animation

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume=True# resume from previous checkpoint?
render = False

# model initialization
D = 210*160 # input dimensionality: 80x80 grid
class Pong:
	def __init__(self,second=False):
		if resume:
			if (second):
				self.model = pickle.load(open('save_pong2.p', 'rb'))

			else:
				self.model = pickle.load(open('save_pong1.p', 'rb'))
		else:
			self.model = {}
			self.model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
			self.model['W2'] = np.random.randn(H) / np.sqrt(H)
		
		self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() } # update buffers that add up gradients over a batch
		self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.items() } # rmsprop memory

	def sigmoid(self,x): 
	  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

	def discount_rewards(self,r):
	  """ take 1D float array of rewards and compute discounted reward """
	  discounted_r = np.zeros_like(r)
	  running_add = 0
	  for t in reversed(range(0, r.size)):
	    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
	    running_add = running_add * gamma + r[t]
	    discounted_r[t] = running_add
	  #print(discounted_r)
	  return discounted_r

	def policy_forward(self,x):
	  h = np.dot(self.model['W1'], x)
	  h[h<0] = 0 # ReLU nonlinearity
	  logp = np.dot(self.model['W2'], h)
	  p = self.sigmoid(logp)
	  return p, h # return probability of taking action 2, and hidden state

	def policy_backward(self,eph, epdlogp):
	  """ backward pass. (eph is array of intermediate hidden states) """
	  dW2 = np.dot(eph.T, epdlogp).ravel()
	  dh = np.outer(epdlogp, self.model['W2'])
	  dh[eph <= 0] = 0 # backpro prelu
	  dW1 = np.dot(dh.T, epx)
	  return {'W1':dW1, 'W2':dW2}

	def display_frames_as_gif(self,frames, filename_gif = None):
	    """
	    Displays a list of frames as a gif, with controls
	    """
	    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
	    patch = plt.imshow(frames[0])
	    plt.axis('off')

	    def animate(i):
	        patch.set_data(frames[i])

	    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
	    if filename_gif: 
	        anim.save(filename_gif, writer = 'imagemagick', fps=50)
	    display(display_animation(anim, default_mode='loop'))
def prepro(I):
  I=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
  I=I.reshape((160,210))
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I= np.rot90(I,k=3)
  I=np.fliplr(I)
  #I = I[2:-2,35:195] # crop
  #I = I[::2,::2] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()



game = pong_game.PongGame()
observation = game.getPresentFrame()
pong1=Pong()
pong2=Pong(second=True)
prev_x = None # used in computing the difference frame
xs,hs,dlogps1,dlogps2,drs1,drs2 = [],[],[],[],[],[]
running_reward = None
reward_sum1 = 0
reward_sum2 = 0
points1=0
points2=0
episode_number = 0
frames=[]
while True:
  #env.render()
  ##frame = env.render(mode = 'rgb_array')
  #frames.append(frame)
  #fig,ax = plt.subplots()
  #firstframe = env.render(mode = 'rgb_array')
  #im = ax.imshow(firstframe)
  #frame = env.render(mode = 'rgb_array')
  #im.set_data(frame)

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  #plt.imshow(np.array(x).reshape(210,160),cmap="gray")
  #plt.show()
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob1, h = pong1.policy_forward(x)
  aprob2, _ = pong2.policy_forward(x) 
  #print(aprob1,aprob2) 
  action1 = 1 if np.random.uniform() < aprob1 else 0 # 1-UP , 0-Down!
  action2 = 1 if np.random.uniform() < aprob2 else 0 # 1-UP , 0-Down!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y1 = 1 if action1 == 1 else 0 # a "fake label"
  y2 = 1 if action2 == 1 else 0 # a "fake label"
  dlogps1.append(y1 - aprob1) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
  dlogps2.append(y2 - aprob2)

  # step the environment and get new measurements
  reward1,reward2, observation = game.getNextFrame(action1,action2)
  if(reward1!=0 and reward2!=0):
  	if(reward1>0):
  		points1+=1
  	if(reward2>0):
  		points2+=1
  	print(points1,points2)
  #reward_t, frame1 = game.getNextFrame(action2)
  reward_sum1 += reward1
  reward_sum2 += reward2

  drs1.append(reward1) # record reward (has to be done after we call step() to get reward for previous action)
  drs2.append(reward2)

  if points1==20 or points2==20: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp1 = np.vstack(dlogps1)
    epdlogp2 = np.vstack(dlogps2)
    epr1 = np.vstack(drs1)
    epr2 = np.vstack(drs2)
    xs,hs,dlogps1,dlogps2,drs1,drs2 = [],[],[],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = pong1.discount_rewards(epr1).astype("float64")
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    #print(discounted_epr,np.mean(discounted_epr).astype("float32"))
    discounted_epr -= np.mean(discounted_epr)
    #print(np.std(discounted_epr))
    discounted_epr /= np.std(discounted_epr)
    #print(discounted_epr)


    epdlogp1 *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
   # print(epdlogp1)

    discounted_epr = pong2.discount_rewards(epr2).astype("float64")
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp2 *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    #print(epdlogp2)

    grad1 = pong1.policy_backward(eph, epdlogp1)
    grad2 = pong2.policy_backward(eph, epdlogp2)
    for k in pong1.model:
    	pong1.grad_buffer[k] += grad1[k] # accumulate grad over batch
    for k in pong2.model:
    	pong2.grad_buffer[k] += grad2[k]

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in pong1.model.items():
        g = pong1.grad_buffer[k] # gradient
        pong1.rmsprop_cache[k] = decay_rate * pong1.rmsprop_cache[k] + (1 - decay_rate) * g**2
        pong1.model[k] += learning_rate * g / (np.sqrt(pong1.rmsprop_cache[k]) + 1e-5)
        pong1.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
        #display_frames_as_gif(frames, filename_gif="manualplay.gif")
        #frames=[]

      for k,v in pong2.model.items():
        g = pong2.grad_buffer[k] # gradient
        pong2.rmsprop_cache[k] = decay_rate * pong2.rmsprop_cache[k] + (1 - decay_rate) * g**2
        pong2.model[k] += learning_rate * g / (np.sqrt(pong2.rmsprop_cache[k]) + 1e-5)
        pong2.grad_buffer[k] = np.zeros_like(v)

    # boring book-keeping
    #running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    #print('resetting env. episode reward total was %f. running mean: %f',%(reward_sum, running_reward))
    #if episode_number % 20 == 0:
    pickle.dump(pong1.model, open('save_pong1.p', 'wb'))
    pickle.dump(pong2.model, open('save_pong2.p', 'wb'))
    print("Saved!!")
    reward_sum1 = 0
    reward_sum2 = 0
    points1=0
    points2=0
    #prev_x = None

  #if reward1 != 0 or reward2 !=0: # Pong has either +1 or -1 reward exactly when game ends.
    #print ('ep %d: game finished, Player1: %f,Player2: %f'%(episode_number, reward1,reward2))