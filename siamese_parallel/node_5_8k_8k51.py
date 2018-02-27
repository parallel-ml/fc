import argparse
import os
import time
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
from collections import deque
from multiprocessing import Queue
from threading import Thread, Lock

import avro.ipc as ipc
import avro.protocol as protocol
import avro.schema as schema
import matplotlib
import numpy as np
import tensorflow as tf
import yaml

import model as ml
import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

matplotlib.use('Agg')

PROTOCOL = protocol.parse(open('resource/image.avpr').read())


class Node(object):
    """ singleton factory with threading safe lock.

    Attributes:
        ip: A dictionary contains Queue of ip addresses for different models type.
        model: loaded models associated to a node.
        extra_model: used by maxpooling layer
        graph: default graph used by Tensorflow
        max_layer_dim: dimension of max pooling layer
        debug: flag for debugging
        max_spatial_input: input at fc layer from spatial CNN
        max_temporal_input: input at fc layer from temporal CNN
        lock: threading lock for safe usage of this class. The lock is used
                for safe models forwarding. If the models is processing input and
                it gets request from other devices, the new request will wait
                until the previous models forwarding finishes.
        name: model name
        total: total time counted
        count: number of frame gets back

    """

    instance = None

    def __init__(self):
        self.ip = dict()
        self.model = None
        self.extra_model = None
        self.graph = tf.get_default_graph()
        self.max_layer_dim = 16
        self.debug = False
        self.max_spatial_input = deque()
        self.max_temporal_input = deque()
        self.lock = Lock()
        self.name = 'unknown'
        self.total = 0
        self.count = 1

    def log(self, step, data=''):
        if self.debug:
            util.step(step, data)

    def acquire_lock(self):
        self.lock.acquire()

    def release_lock(self):
        self.lock.release()

    def timer(self, interval):
        self.total += interval
        print '{:s}: {:.3f}'.format(self.name, self.total / self.count)
        self.count += 1

    @classmethod
    def create(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance


class Responder(ipc.Responder):
    """ responder called by handler when got request """

    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        """ process response

        invoke handles the request and get response for the request. This is the key
        of each node. All models forwarding and output redirect are done here.

        Args:
            msg: meta data
            req: contains data packet

        Returns:
            a string of data

        Raises:
            AvroException: if the data does not have correct syntac defined in Schema

        """
        node = Node.create()
        node.acquire_lock()

        if msg.name == 'forward':
            try:
                with node.graph.as_default():
                    bytestr = req['input']
                    if req['next'] == 'spatial':
                        node.log('get spatial request')
                        X = np.fromstring(bytestr, np.uint8).reshape(12, 16, 3)
                        node.model = ml.load_spatial() if node.model is None else node.model
                        output = node.model.predict(np.array([X]))
                        node.log('finish spatial forward')
                        Thread(target=self.send, args=(output, 'fc_1', 'spatial')).start()

                    elif req['next'] == 'temporal':
                        node.log('get temporal request')
                        X = np.fromstring(bytestr, np.float32).reshape(12, 16, 20)
                        node.model = ml.load_temporal() if node.model is None else node.model
                        output = node.model.predict(np.array([X]))
                        node.log('finish temporal forward')
                        Thread(target=self.send, args=(output, 'fc_1', 'temporal')).start()

                    elif req['next'] == 'fc_1':
                        tag = req['tag']
                        X = np.fromstring(bytestr, np.float32)
                        X = X.reshape(1, X.size)
                        if tag == 'spatial':
                            node.max_spatial_input.append(X)
                        else:
                            node.max_temporal_input.append(X)
                        node.log('get head request', 'spatial: %d temporal: %d' % (
                            len(node.max_spatial_input), len(node.max_temporal_input)))
                        if len(node.max_spatial_input) < node.max_layer_dim or len(
                                node.max_temporal_input) < node.max_layer_dim:
                            node.release_lock()
                            return

                        # pop extra frame due to transmitting delay
                        while len(node.max_spatial_input) > node.max_layer_dim:
                            node.max_spatial_input.popleft()
                        while len(node.max_temporal_input) > node.max_layer_dim:
                            node.max_temporal_input.popleft()
                        node.model = ml.load_maxpool(input_shape=(node.max_layer_dim, 256),
                                                     N=node.max_layer_dim) if node.model is None else node.model
                        # concatenate inputs from spatial and temporal
                        # ex: (1, 256) + (1, 256) = (2, 256)
                        s_input = np.concatenate(node.max_spatial_input)
                        t_input = np.concatenate(node.max_temporal_input)
                        s_output = node.model.predict(np.array([s_input]))
                        t_output = node.model.predict(np.array([t_input]))
                        output = np.concatenate([s_output, t_output], axis=1)
                        output = output.reshape(output.size)

                        # start forward at head node
                        node.extra_model = ml.load_fc_1(input_shape=7680,
                                                        output_shape=8192) if node.extra_model is None else node.extra_model
                        output = node.extra_model.predict(np.array([output]))

                        node.log('finish max pooling')

                        # pop least recent frame from deque
                        node.max_spatial_input.popleft()
                        node.max_temporal_input.popleft()
                        node.log('finish fc_1 forward')
                        Thread(target=self.send, args=(output, 'fc_2', '')).start()

                    else:
                        X = np.fromstring(bytestr, np.float32)
                        X = X.reshape(X.size)
                        node.log('get fc_2 layer request', X.shape)
                        node.model = ml.load_fc_23(input_shape=8192) if node.model is None else node.model
                        output = node.model.predict(np.array([X]))
                        node.log('finish fc_2 forward')
                        Thread(target=self.send, args=(output, 'initial', '')).start()

                node.release_lock()
                return

            except Exception, e:
                node.log('Error', e.message)
        else:
            raise schema.AvroException('unexpected message:', msg.getname())

    def send(self, X, name, tag):
        """ send data to other devices

        Send data to other devices. The data packet contains data and models name.
        Ip address of next device pop from Queue of a ip list.

        Args:
             X: numpy array
             name: next device models name
             tag: mark the current layer label

        """
        node = Node.create()
        queue = node.ip[name]
        address = queue.get()

        port = 9999 if name == 'initial' else 12345
        client = ipc.HTTPTransceiver(address, port)
        requestor = ipc.Requestor(PROTOCOL, client)

        node.name = name

        data = dict()
        data['input'] = X.tostring()
        data['next'] = name
        data['tag'] = tag
        node.log('finish assembly')
        start = time.time()
        requestor.request('forward', data)
        end = time.time()
        node.timer(end - start)

        node.log('node gets request back')
        client.close()
        queue.put(address)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """ handle request from other devices.

        do_POST is automatically called by ThreadedHTTPServer. It creates a new
        responder for each request. The responder generates response and write
        response to data sent back.

        """
        self.responder = Responder()
        call_request_reader = ipc.FramedReader(self.rfile)
        call_request = call_request_reader.read_framed_message()
        resp_body = self.responder.respond(call_request)
        self.send_response(200)
        self.send_header('Content-Type', 'avro/binary')
        self.end_headers()
        resp_writer = ipc.FramedWriter(self.wfile)
        resp_writer.write_framed_message(resp_body)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """handle requests in separate thread"""


def main(cmd):
    node = Node.create()

    node.max_layer_dim = cmd.max_dim
    node.debug = cmd.debug

    # read ip resources from config file
    with open('resource/ip') as file:
        address = yaml.safe_load(file)
        node.ip['fc'] = Queue()
        node.ip['maxpool'] = Queue()
        node.ip['initial'] = Queue()
        node.ip['fc_1'] = Queue()
        node.ip['fc_2'] = Queue()
        address = address['5_8k_8k51']
        for addr in address['fc_1']:
            if addr == '#':
                break
            node.ip['fc_1'].put(addr)
        for addr in address['fc_2']:
            if addr == '#':
                break
            node.ip['fc_2'].put(addr)
        for addr in address['initial']:
            if addr == '#':
                break
            node.ip['initial'].put(addr)

    server = ThreadedHTTPServer(('0.0.0.0', 12345), Handler)
    server.allow_reuse_address = True
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_dim', metavar='\b', action='store', default=16, type=int,
                        help='Choose maxpooling layer input dimension')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Set to debug mode')
    cmd = parser.parse_args()
    main(cmd)
