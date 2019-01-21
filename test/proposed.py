import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import tensorflow as tf
import fixed_env as env
import a3c
import load_trace
import matplotlib.pyplot as plt
import time


S_INFO = 12  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 5  # take how many frames in the past
A_DIM = 10
ACTOR_LR_RATE = 0.001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 200.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log_sim_rl'
TRAIN_SEQ_LEN = 100
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = './models/nn_model_ep_9100.ckpt'
DEFAULT_SELECTION = 0
age = np.zeros((A_DIM,10000000))
#gamma = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]#, 1,1,0.9,0.85,0.8,0.7,0.6,0.6,0.6,0.5]
gamma = [1,1,0.9,0.85,0.8,0.7,0.6,0.6,0.6,0.5]
violation = np.zeros((A_DIM,1))
violation_n_k = np.zeros((A_DIM,10000000))
violation_hist = np.zeros((A_DIM,1))
lamba = 1000
mu = 0.0#10*lamba
#tau = [50,100,150,200,250,300,350,400,450,500]
tau = [30,50,70,90,110,130,150,170,190,210]
PACKET_SIZE = [50,100,150,200,250,300,350,400,450,500]
#PACKET_SIZE = [500,550,600,650,700,750,800,850,900,950]
epsilon = [0.001,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.005]
hamza = np.zeros((A_DIM,1))
anis = np.zeros((A_DIM,10000000))
j = np.zeros((A_DIM,1))
expected_age  = np.zeros((A_DIM,1))
expected_age_n = np.zeros((A_DIM,1))
exp_queue = []

def main():

    np.random.seed(RANDOM_SEED)

    assert len(PACKET_SIZE) == A_DIM

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace()

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        time_stamp = 0

        sensor_selection = DEFAULT_SELECTION
        
        action_vec = np.zeros(A_DIM)
        action_vec[sensor_selection] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
	video_count = 0
        k = 0
	sum_age = 0
	sum_violation = 0
	violation_n = np.zeros((A_DIM,1))
        while k < 30000:  # serve video forever
	    
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, video_chunk_size = net_env.get_video_chunk(sensor_selection)

            

            #time_stamp += delay  # in ms
            #time_stamp += sleep_time  # in ms

	    
	    
	    #for n in range(0,A_DIM):
	    #    violation[n] = 0
	    #	if n == sensor_selection:
	#	    age[n,k] = delay
	#	else:
	#	    age[n,k] = age[n,k-1] + delay
	#	if age[n,k] > tau[n]:
	#	    violation[n] += 1
	#   sum_age = np.sum(age[:,:])
	#    sum_violation = np.sum(violation)
        #    expected_age=sum_age/(k*A_DIM)

	    sum_age_before = np.sum(age[:,k])
            current_violation = 0
	    for n in range(0,A_DIM):
		
	        #for k in range (1,TRAIN_SEQ_LEN):
		if n == sensor_selection:
		    #print (j)
		    #time.sleep(2)
		    dummy = int(j[n])
                    j[n] += 1
		    age[n,k] = delay
		    anis[n,dummy]= age[n,k]
                    
		    #violation[n] = 0
                    
		else:
		    age[n,k] = age[n,k-1] + delay
		    dummy = int(j[n])
                    anis[n,dummy]= age[n,k]
		if age[n,k] > tau[n]:
		    violation[n] += 1
                    current_violation = current_violation+(10-n/10)
                    violation_n_k[n,k] += 1

	    
            #print violation_n
            #time.sleep(2) 
            for n in range(0,A_DIM):
		#expected_age[n] = gamma[n]*np.sum((anis[n,:int(j[n])+1])/(int(j[n])+1))
		expected_age_n[n]=np.sum(age[n,:])/((k+1))
                if violation_n[n] > epsilon[n]:
                    hamza[n] = 1
		else:
		    hamza[n] = 0

            expected_age = np.sum(expected_age_n[:])/A_DIM

	    #sum_age += np.sum(age)
            #reward = (-np.sum(age[:,k]) - lamba*np.sum(violation_n_k[:,k]) - mu*np.sum(hamza[:]))/100
            reward = (-np.sum(age[:,k]) - lamba*current_violation - mu*np.sum(hamza[:]))/100
	    sum_age += np.sum(age)
	    if k == 29999:
		for n in range(0,A_DIM):
	    	    violation_n[n] = 1000*(10-n/10)*violation[n]/(k+1)
		sum_age = sum_age/((k+1)*A_DIM)
		sum_violation = np.sum(violation_n)
	   	print(sum_age+sum_violation)
		print(100*violation[:]/(k+1))
		print(expected_age_n[:])
		

            r_batch.append(reward)
            log_file.write(str(time_stamp) + '\t' +
                           str(PACKET_SIZE[sensor_selection]) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\t' +
			   str(age[0,k]) + '\t' +
			   str(age[1,k]) + '\t'+
			   str(age[2,k]) + '\t'+
			   str(age[3,k]) + '\t' +
			   str(age[4,k]) + '\t' +
			   str(age[5,k]) + '\t' +
			   str(age[6,k]) + '\t'+
			   str(age[7,k]) + '\t' +
			   str(age[8,k]) + '\t' +
			   str(age[9,k]) + '\n')
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

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
	   
	     # compute action probability vector
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            sensor_selection = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
	    
	    # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))
	    time_stamp += 1
            # log time_stamp, bit_rate, buffer_size, reward
           
		

	    #if end_of_video:
                
            #    del s_batch[:]
            #    del a_batch[:]
            #    del r_batch[:]
            #    del entropy_record[:]
		#k = 0
		#for n in range(0,A_DIM):
		#    violation[n] = 0
		#    age[n,:] = 0
		#sensor_selection = DEFAULT_SELECTION

            #log_file.write('\n')  # so that in the log we know where video ends
            s_batch.append(state)

            action_vec = np.zeros(A_DIM)
            action_vec[sensor_selection] = 1
            a_batch.append(action_vec)
            #log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
		#log_file = open(log_path, 'wb')
            k += 1

if __name__ == '__main__':
    main()
