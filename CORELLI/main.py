import d3rlpy
from d3rlpy.datasets import get_cartpole
from d3rlpy.algos import DiscreteCQL
from sklearn.model_selection import train_test_split

dataset, env = get_cartpole() 
cql = DiscreteCQL(use_gpu = True)    # instantiate discrete CQL algrithm 

train_episodes, test_episodes = train_test_split(dataset, test_size = 0.2)

#cql.fit(dataset, n_steps = 10000)   # train (offline)
cql.fit(train_episodes, eval_episodes=test_episodes, n_steps = 10000)
# actions = cql.predict(x)    # generate actions



# NEW STUFF GOES BELOW


import glob
import io
import base64

from gym.wrappers import Monitor
from IPython.display import HTML
from IPython import display as ipythondisplay
from d3rlpy.metrics.scorer import evaluate_on_environment
from pyvirtualdisplay import Display

# start virtual display
display = Display(visible=0, size=(1400, 900))
display.start()

# play recorded video
def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''
            <video alt="test" autoplay loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
            </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")
    

# wrap Monitor wrapper
env = Monitor(env, './video', force=True)

# evaluate
evaluate_on_environment(env)(cql)


show_video()