from Environment import EnvironmentTSN
from Environment_A import EnvironmentA
from Environment_B import EnvironmentB
from dqn import DQNAgent
from dqn_A import DQNAgentA
from dqn_B import DQNAgentB
from config import BACKGROUND_STREAMS, VNF_LENGTH, VNF_PERIOD, VNF_DELAY, TRAINING_STEPS, ROUTE_EVALUATION_EPISODES, \
    CUSTOM_EVALUATION_EPISODES, MONITOR_TRAINING

main_id = 'dist_3'

while True:
    try:
        option = int(input('[*] Select the option you want (0 = train A, 1 = train B, 2 = evaluate A routes, 3 = evaluate A custom, 4 = evaluate B routes, 5 = evaluate B custom, 6 = evaluate A&B custom): '))
        if 0 <= option <= 6:
            break
        else:
            print('[!] Option not valid! Try again...')
    except ValueError:
        print('[!] Expected to introduce a number! Try again...')

if option == 0:
    ENV = EnvironmentA(main_id, False)

    agent = DQNAgentA(
        env=ENV,
        log_file_id=main_id
    )

    max_steps = TRAINING_STEPS
    agent.logger.info('[I] Chose training model A')
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
    ENV = EnvironmentB(main_id, False)

    agent = DQNAgentB(
        env=ENV,
        log_file_id=main_id
    )

    max_steps = TRAINING_STEPS
    agent.logger.info('[I] Chose training model B')
    agent.logger.info('[I] Settings: time_steps = ' + str(max_steps) +
                      ' | epsilon = ' + str(agent.epsilon_decay) + ' | background flows = ' + str(BACKGROUND_STREAMS) +
                      ' | replay_buffer = ' + str(agent.replay_buffer_size.max_size) +
                      ' | batch size = ' + str(agent.batch_size) +
                      ' | target update = ' + str(agent.update_target_every_steps) + ' | gamma = ' + str(agent.gamma) +
                      ' | learning rate = ' + str(agent.learning_rate) + ' | tau = ' + str(agent.tau) +
                      ' | vnf_len = ' + str(VNF_LENGTH) + ' | vnf_prd = ' + str(VNF_PERIOD) + ' | vnf_delay = ' + str(VNF_DELAY))
    agent.logger.info('[I] Extra info: ' + str(input('[*] Introduce any other setting data (just to print it): ')))
    agent.train(max_steps=max_steps, monitor_training=MONITOR_TRAINING, plotting_interval=max_steps)

elif option == 2:
    ENV = EnvironmentA(main_id, True)

    agent = DQNAgentA(
        env=ENV,
        log_file_id=main_id
    )

    agent.logger.info('[I] Chose evaluating custom model A routes')
    routing_model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(routing_model_name)
    agent.evaluate_routes(ENV, ROUTE_EVALUATION_EPISODES)

elif option == 3:
    ENV = EnvironmentA(main_id, True)

    agent = DQNAgentA(
        env=ENV,
        log_file_id=main_id
    )

    agent.logger.info('[I] Chose evaluating custom model A')
    model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(model_name)
    agent.evaluate(ENV, CUSTOM_EVALUATION_EPISODES)

elif option == 4:
    ENV = EnvironmentB(main_id, True)

    agent = DQNAgentB(
        env=ENV,
        log_file_id=main_id
    )

    agent.logger.info('[I] Chose evaluating custom model B routes')
    routing_model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(routing_model_name)
    agent.evaluate_routes(ENV, ROUTE_EVALUATION_EPISODES)

elif option == 5:
    ENV = EnvironmentB(main_id, True)

    agent = DQNAgentB(
        env=ENV,
        log_file_id=main_id
    )

    agent.logger.info('[I] Chose evaluating custom model B')
    model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(model_name)
    agent.evaluate(ENV, CUSTOM_EVALUATION_EPISODES)

elif option == 6:
    ENV = EnvironmentTSN(main_id, True)

    agent = DQNAgent(
        env=ENV,
        log_file_id=main_id
    )

    print('[I] Chose evaluating custom models')
    model_name = input('[*] Introduce the name of the model A: ')
    agent.agent_A.load_custom_model(model_name)
    model_name = input('[*] Introduce the name of the model B: ')
    agent.agent_B.load_custom_model(model_name)
    agent.evaluate(ENV, CUSTOM_EVALUATION_EPISODES)
