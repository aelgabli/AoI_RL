import os
import logging
import numpy as np
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import env
import a3c
import load_trace
import time


S_INFO = 12  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 5  # take how many frames in the past
A_DIM = 10
ACTOR_LR_RATE = 0.001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 1
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 200.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_SELECTION = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'
#NN_MODEL = './results/nn_model_ep_6600.ckpt'
NN_MODEL = None
age = np.zeros((A_DIM,10000000))
violation = np.zeros((A_DIM,1))
lamba = 1000
tau = [30,50,70,90,110,130,150,170,190,210]
PACKET_SIZE = [50,100,150,200,250,300,350,400,450,500]
expected_age  = np.zeros((A_DIM,1))
expected_age_n = np.zeros((A_DIM,1))

def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with tf.Session() as sess, open(LOG_FILE + '_test', 'wb') as test_log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)
	

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0

        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in xrange(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0 
	    
            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in xrange(NUM_AGENTS):
		s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

		
		actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)
		
                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            for i in xrange(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()
	    if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
	        logging.info("Model saved in file: " + save_path)
		print "MODEL READY"
                #testing(epoch, 
                #    SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", 
                #    test_log_file)


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue):

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    with tf.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        sensor_selection = DEFAULT_SELECTION
        
        action_vec = np.zeros(A_DIM)
        action_vec[sensor_selection] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
	k = 0
        sum_age = 0
	sum_violation = 0	

        while True:  # experience video streaming forever
	    
            
            
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, video_chunk_size = net_env.get_video_chunk(sensor_selection)

            max_age = (age[:,k]).argmax()

	    sum_age_before = np.sum(age[:,k])
            current_violation = 0
	    for n in range(0,A_DIM):
		
	        #for k in range (1,TRAIN_SEQ_LEN):
		if n == sensor_selection:
		    age[n,k] = delay
		    
		else:
		    age[n,k] = age[n,k-1] + delay

		if age[n,k] > tau[n]:
                    current_violation +=1

            for n in range(0,A_DIM):
		expected_age_n[n]=np.sum(age[n,:])/((k+1))


            expected_age = np.sum(expected_age_n[:])/A_DIM

            reward = (-np.sum(age[:,k]) - lamba*current_violation)/100
	    
            r_batch.append(reward)
	    
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms

            state[0, -1] = float(age[0,k])/M_IN_K
            state[1, -1] = float(age[1,k])/M_IN_K 
            state[2, -1] = float(age[2,k])/M_IN_K 
            state[3, -1] = float(age[3,k])/M_IN_K 
            state[4, -1] = float(age[4,k])/M_IN_K 
            state[5, -1] = float(age[5,k])/M_IN_K 
            state[6, -1] = float(age[6,k])/M_IN_K
            state[7, -1] = float(age[7,k])/M_IN_K 
            state[8, -1] = float(age[8,k])/M_IN_K 
            state[9, -1] = float(age[9,k])/M_IN_K 
	    #state[10, -1] = float(PACKET_SIZE[0])/float(PACKET_SIZE[9])
            #state[11, -1] = float(PACKET_SIZE[1])/float(PACKET_SIZE[9]) 
            #state[12, -1] = float(PACKET_SIZE[2])/float(PACKET_SIZE[9]) 
            #state[13, -1] = float(PACKET_SIZE[3])/float(PACKET_SIZE[9]) 
            #state[14, -1] = float(PACKET_SIZE[4])/float(PACKET_SIZE[9]) 
            #state[15, -1] = float(PACKET_SIZE[5])/float(PACKET_SIZE[9]) 
            #state[16, -1] = float(PACKET_SIZE[6])/float(PACKET_SIZE[9])
            #state[17, -1] = float(PACKET_SIZE[7])/float(PACKET_SIZE[9]) 
            #state[18, -1] = float(PACKET_SIZE[8])/float(PACKET_SIZE[9]) 
            #state[19, -1] = float(PACKET_SIZE[9])/float(PACKET_SIZE[9])
	    state[10, -1] = float(delay)/100
	    state[11, -1] = float(PACKET_SIZE[sensor_selection])/(100*float(delay)*float(PACKET_SIZE[9]))
            
	    log_file.write(str(time_stamp) + '\t' +
			   str(reward) + '\t' + str(age[:,k]) + '\t' + str(expected_age_n) + '\n')
            log_file.flush()
            
            # compute action probability vector
	    action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            sensor_selection = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            
	    entropy_record.append(a3c.compute_entropy(action_prob[0]))
	    time_stamp += 1
            
            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN: #or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
			       True,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]
		

            s_batch.append(state)

            action_vec = np.zeros(A_DIM)
            action_vec[sensor_selection] = 1
            a_batch.append(action_vec)
            k += 1


def main():

    np.random.seed(RANDOM_SEED)
    assert len(PACKET_SIZE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in xrange(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))


    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()
    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in xrange(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in xrange(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
