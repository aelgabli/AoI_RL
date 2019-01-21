import numpy as np
import time
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 0.001  # sec
#PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = './video_size_'
probability_drop = 0.1
PACKET_SIZE = [50,100,150,200,250,300,350,400,450,500]


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        #self.video_size = {}  # in bytes
        #for bitrate in xrange(BITRATE_LEVELS):
        #    self.video_size[bitrate] = []
        #    with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
        #        for line in f:
        #            self.video_size[bitrate].append(int(line.split()[0]))

    def get_video_chunk(self, quality):
	drop = np.random.randint(1, 10)
        
        #video_chunk_size = PACKET_SIZE[quality]
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes
        while drop <= 1:
            video_chunk_size = PACKET_SIZE[quality]
            video_chunk_counter_sent = 0  # in bytes
            while True:  # download video chunk over mahimahi
                throughput = self.cooked_bw[self.mahimahi_ptr] \
                             * B_IN_MB / BITS_IN_BYTE
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
	    
                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent + packet_payload > video_chunk_size:

                    fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    delay += fractional_time
                    self.last_mahimahi_time += fractional_time
                    assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                    break

                video_chunk_counter_sent += packet_payload
                delay += duration
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

            #delay *= MILLISECONDS_IN_SECOND
            delay += LINK_RTT

	    # add a multiplicative noise to the delay
	    delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        
            sleep_time = 0
            drop = np.random.randint(1, 10)
            #print drop
            #time.sleep(1)
        
        video_chunk_size = PACKET_SIZE[quality]
        video_chunk_counter_sent = 0  # in bytes
        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time
	    
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay += LINK_RTT
        delay *= MILLISECONDS_IN_SECOND
        

	# add a multiplicative noise to the delay
	delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        
        sleep_time = 0
        

        self.video_chunk_counter += 1
       

        return delay, \
            sleep_time, \
            video_chunk_size, \
