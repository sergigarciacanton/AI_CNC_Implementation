from Environment import EnvironmentTSN
from Environment_A import EnvironmentA
from Environment_B import EnvironmentB
from dqn import DQNAgent
from dqn_distributed_A import DQNAgentA
from dqn_distributed_B import DQNAgentB
from config_distributed import BACKGROUND_STREAMS, VNF_LENGTH, VNF_PERIOD, VNF_DELAY


main_id = 'dist_1'

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
    ENV = EnvironmentA(main_id)

    agent = DQNAgentA(
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

    max_steps = 200000
    agent.logger.info('[I] Chose training model A')
    agent.logger.info('[I] Settings: time_steps = ' + str(max_steps) +
                      ' | epsilon = ' + str(agent.epsilon_decay) + ' | background flows = ' + str(BACKGROUND_STREAMS) +
                      ' | replay_buffer = ' + str(agent.replay_buffer_size.max_size) +
                      ' | batch size = ' + str(agent.batch_size) +
                      ' | target update = ' + str(agent.update_target_every_steps) + ' | gamma = ' + str(agent.gamma) +
                      ' | learning rate = ' + str(agent.learning_rate) + ' | tau = ' + str(agent.tau) + 
                      ' | vnf_len = ' + str(VNF_LENGTH) + ' | vnf_prd = ' + str(VNF_PERIOD) + ' | vnf_delay = ' + str(VNF_DELAY))
    agent.logger.info('[I] Extra info: ' + str(input('[*] Introduce any other setting data (just to print it): ')))
    agent.train(max_steps=max_steps, monitor_training=10000, plotting_interval=max_steps)

elif option == 1:
    ENV = EnvironmentB(main_id)

    agent = DQNAgentB(
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

    max_steps = 200000
    agent.logger.info('[I] Chose training model B')
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
    ENV = EnvironmentA(main_id)

    agent = DQNAgentA(
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

    agent.logger.info('[I] Chose evaluating custom model A routes')
    routing_model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(routing_model_name)
    agent.evaluate_routes(ENV, 100)

elif option == 3:
    ENV = EnvironmentA(main_id)

    agent = DQNAgentA(
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

    agent.logger.info('[I] Chose evaluating custom model A')
    model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(model_name)
    agent.evaluate(ENV, 1000)

elif option == 4:
    ENV = EnvironmentB(main_id)

    agent = DQNAgentB(
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

    agent.logger.info('[I] Chose evaluating custom model B routes')
    routing_model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(routing_model_name)
    agent.evaluate_routes(ENV, 100)

elif option == 5:
    ENV = EnvironmentB(main_id)

    agent = DQNAgentB(
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

    agent.logger.info('[I] Chose evaluating custom model B')
    model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(model_name)
    agent.evaluate(ENV, 1000)

elif option == 6:
    ENV = EnvironmentTSN(main_id)

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

    print('[I] Chose evaluating custom models')
    model_name = input('[*] Introduce the name of the model A: ')
    agent.agent_A.load_custom_model(model_name)
    # agent.agent_A.load_custom_model('/home/upc_ai_vecn/Documents/AI_CNC_Implementation/old_models/test_distributed/model_202409131936_246_A.pt')
    model_name = input('[*] Introduce the name of the model B: ')
    agent.agent_B.load_custom_model(model_name)
    # agent.agent_B.load_custom_model('/home/upc_ai_vecn/Documents/AI_CNC_Implementation/old_models/test_distributed/model_202409141009_298_B.pt')
    agent.evaluate(ENV, 1000)
