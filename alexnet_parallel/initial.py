import time
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
from collections import deque
from multiprocessing import Queue
from threading import Thread

import avro.ipc as ipc
import avro.protocol as protocol
import avro.schema as schema
import cv2
import numpy as np
import yaml

PROTOCOL = protocol.parse(open('resource/image.avpr').read())


class Initializer:
    """ singleton factory for initializer node

    Attributes:
        queue: Queue for storing block1 models devices
        start: start time of getting a frame
        count: total number of frames gets back
        node_total: total layerwise time
        node_count: total layerwise frame count

    """
    instance = None

    def __init__(self):
        self.queue = Queue()
        self.start = 0.0
        self.count = 0
        self.node_total = 0
        self.node_count = 1

    def timer(self):
        if self.count == 0:
            self.start = time.time()
        else:
            print 'total time: {:.3f} sec'.format((time.time() - self.start) / self.count)
        self.count += 1

    def node_timer(self, mode, interval):
        self.node_total += interval
        print '{:s}: {:.3f}'.format(mode, self.node_total / self.node_count)
        self.node_count += 1

    @classmethod
    def create_init(cls):
        if cls.instance is None:
            cls.instance = Initializer()
        return cls.instance


def send_request(bytestr, mode, tag=''):
    init = Initializer.create_init()
    queue = init.queue

    addr = queue.get()
    client = ipc.HTTPTransceiver(addr, 12345)
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    data['input'] = bytestr
    data['next'] = mode
    data['tag'] = tag

    start = time.time()
    requestor.request('forward', data)
    end = time.time()

    init.node_timer(mode, end - start)

    client.close()
    queue.put(addr)


def master():
    """ master function for real time video.

    A basic while loop gets one frame at each time. It appends a frame to deque
    every time and pop the least recent one if the length > maximum.
    """
    init = Initializer.create_init()
    while True:
        # current frame
        ret, frame = 'unknown', np.random.rand(220, 220, 3) * 255
        frame = frame.astype(dtype=np.uint8)
        Thread(target=send_request, args=(frame.tobytes(), 'block1', 'initial')).start()
        time.sleep(0.03)


class Responder(ipc.Responder):
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
        if msg.name == 'forward':
            init = Initializer.create_init()
            try:
                init.timer()
                return
            except Exception, e:
                print 'Error', e.message
        else:
            raise schema.AvroException('unexpected message:', msg.getname())


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


def main():
    init = Initializer.create_init()
    # read ip resources from config file
    with open('resource/ip') as file:
        address = yaml.safe_load(file)
        address = address['node_4']
        for addr in address['block1']:
            if addr == '#':
                break
            init.queue.put(addr)

    server = ThreadedHTTPServer(('0.0.0.0', 9999), Handler)
    server.allow_reuse_address = True
    Thread(target=server.serve_forever, args=()).start()

    master()


if __name__ == '__main__':
    main()
