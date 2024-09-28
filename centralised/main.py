from Environment import EnvironmentTSN
from dqn import DQNAgent
from config import BACKGROUND_STREAMS, VNF_LENGTH, VNF_PERIOD, VNF_DELAY, TRAINING_STEPS, MONITOR_TRAINING, \
    ROUTE_EVALUATION_EPISODES, CUSTOM_EVALUATION_EPISODES

main_id = 'cent_1'

while True:
    try:
        option = int(input('[*] Select the option you want (0 = train, 1 = evaluate routes, 2 = evaluate custom): '))
        if option == 0 or option == 1 or option == 2:
            break
        else:
            print('[!] Option not valid! Try again...')
    except ValueError:
        print('[!] Expected to introduce a number! Try again...')

if option == 0:
    ENV = EnvironmentTSN(main_id, False)

    agent = DQNAgent(
        env=ENV,
        log_file_id=main_id
    )

    max_steps = TRAINING_STEPS
    agent.logger.info('[I] Chose training model')
    agent.logger.info('[I] Settings: time_steps = ' + str(max_steps) +
                      ' | epsilon = ' + str(agent.epsilon_decay) + ' | background flows = ' + str(BACKGROUND_STREAMS) +
                      ' | replay_buffer = ' + str(agent.replay_buffer_size.max_size) +
                      ' | batch size = ' + str(agent.batch_size) +
                      ' | target update = ' + str(agent.update_target_every_steps) + ' | gamma = ' + str(agent.gamma) +
                      ' | learning rate = ' + str(agent.learning_rate) + ' | tau = ' + str(agent.tau) + 
                      ' | vnf_len = ' + str(VNF_LENGTH) + ' | vnf_prd = ' + str(VNF_PERIOD) + ' | vnf_delay = ' + str(VNF_DELAY))
    agent.logger.info('[I] Extra info: ' + str(input('[*] Introduce any other setting data (just to print it): ')))
    agent.train(max_steps=max_steps, monitor_training=MONITOR_TRAINING, plotting_interval=max_steps)

elif option == 1:
    ENV = EnvironmentTSN(main_id, True)

    agent = DQNAgent(
        env=ENV,
        log_file_id=main_id
    )

    agent.logger.info('[I] Chose evaluating custom model routes')
    routing_model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(routing_model_name)
    agent.evaluate_routes(ENV, ROUTE_EVALUATION_EPISODES)

elif option == 2:
    ENV = EnvironmentTSN(main_id, True)

    agent = DQNAgent(
        env=ENV,
        log_file_id=main_id
    )

    agent.logger.info('[I] Chose evaluating custom model')
    model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(model_name)
    agent.evaluate(ENV, CUSTOM_EVALUATION_EPISODES)
