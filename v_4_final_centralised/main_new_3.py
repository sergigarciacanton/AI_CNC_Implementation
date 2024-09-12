from Environment_new import EnvironmentTSN
import time
from dqn_new import DQNAgent
from config_new import BACKGROUND_STREAMS, VNF_LENGTH, VNF_PERIOD, VNF_DELAY


main_id = 'new_3'
ENV = EnvironmentTSN(main_id)

while ENV.ready is False:
    time.sleep(1)

agent = DQNAgent(
    env=ENV,
    replay_buffer_size=1000000,
    batch_size=6,
    target_update=400,
    epsilon_decay=1 / 80000,
    seed=None,
    max_epsilon=1.0,
    min_epsilon=0.0,
    gamma=0.999,
    learning_rate=0.0001,
    tau=0.85,
    log_file_id=main_id
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
    max_steps = 200000
    agent.logger.info('[I] Chose training model')
    agent.logger.info('[I] Settings: time_steps = ' + str(max_steps) +
                      ' | epsilon = ' + str(agent.epsilon_decay) + ' | background flows = ' + str(BACKGROUND_STREAMS) +
                      ' | replay_buffer = ' + str(agent.replay_buffer_size.max_size) +
                      ' | batch size = ' + str(agent.batch_size) +
                      ' | target update = ' + str(agent.update_target_every_steps) + ' | gamma = ' + str(agent.gamma) +
                      ' | learning rate = ' + str(agent.learning_rate) + ' | tau = ' + str(agent.tau) +
                      ' | vnf_len = ' + str(VNF_LENGTH) + ' | vnf_prd = ' + str(VNF_PERIOD) + ' | vnf_delay = ' + str(VNF_DELAY))
    agent.logger.info('[I] Extra info: ' + str(input('[*] Introduce any other setting data (just to print it): ')))
    agent.train(max_steps=max_steps, monitor_training=10000, plotting_interval=max_steps)

elif option == 2:
    agent.logger.info('[I] Chose evaluating custom model')
    model_name = input('[*] Introduce the name of the routing model: ')
    agent.load_custom_model(model_name)
    agent.evaluate(ENV, 1000)
