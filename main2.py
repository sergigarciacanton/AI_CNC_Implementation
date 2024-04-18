from Environment import EnvironmentTSN
import time
from dqn import DQNAgent


ENV = EnvironmentTSN()

while ENV.ready is False:
    time.sleep(1)

agent = DQNAgent(
    env=ENV,
    replay_buffer_size=1000000,
    batch_size=8,
    target_update=1000,
    epsilon_decay=1 / 40000,
    seed=None,
    max_epsilon=1.0,
    min_epsilon=0.0,
    gamma=0.999,
    learning_rate=0.0001,
    tau=0.85
)

while True:
    try:
        option = int(input('[*] Select the option you want (0 = train, 1 = evaluate best, 2 = evaluate custom): '))
        if option == 0 or option == 1 or option == 2:
            break
        else:
            print('[!] Option not valid! Try again...')
    except ValueError:
        print('[!] Expected to introduce a number! Try again...')

if option == 0:
    print('[I] Chose training model')
    agent.train(max_steps=100000, monitor_training=10000, plotting_interval=20000)

elif option == 1:
    print('[I] Chose evaluating best model')
    agent.load_model()
    agent.evaluate(ENV, 100)

elif option == 2:
    print('[I] Chose evaluating custom model')
    model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(model_name)
    agent.evaluate(ENV, 1000)
