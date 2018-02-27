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
        spatial_q: Queue for storing spatial models devices
        temporal_q: Queue for storing temporal models devices
        flows: deque for storing fixed size frames
        start: start time of getting back a frame
        count: total number of frames gets back
        sp_total: total time of spatial layer
        sp_count: total count of spatial feedback
        tmp_total: total time of temporal layer
        tmp_count: total count of temporal feedback

    """
    instance = None

    def __init__(self):
        self.spatial_q = Queue()
        self.temporal_q = Queue()
        self.flows = deque()
        self.start = 0.0
        self.count = 0
        self.sp_total = 0
        self.sp_count = 1
        self.tmp_total = 0
        self.tmp_count = 1

    def timer(self):
        if self.count == 0:
            self.start = time.time()
        else:
            print 'total time: {:.3f} sec'.format((time.time() - self.start) / self.count)
        self.count += 1

    def node_timer(self, mode, interval):
        if mode == 'spatial':
            self.sp_total += interval
            print '{:s}: {:.3f}'.format(mode, self.sp_total / self.sp_count)
            self.sp_count += 1
        else:
            self.tmp_total += interval
            print '{:s}: {:.3f}'.format(mode, self.tmp_total / self.tmp_count)
            self.tmp_count += 1

    @classmethod
    def create_init(cls):
        if cls.instance is None:
            cls.instance = Initializer()
        return cls.instance


def send_request(bytestr, mode='spatial'):
    init = Initializer.create_init()
    queue = init.spatial_q if mode == 'spatial' else init.temporal_q

    addr = queue.get()
    client = ipc.HTTPTransceiver(addr, 12345)
    requestor = ipc.Requestor(PROTOCOL, client)

    data = dict()
    data['input'] = bytestr
    data['next'] = mode
    data['tag'] = ''

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
    # for previous frame used
    frame0 = None
    while True:
        # current frame
        ret, frame = 'unknown', np.random.rand(12, 16, 3) * 255
        frame = frame.astype(dtype=np.uint8)
        image = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame0 is not None:
            # append new 1-1 optical frame into deque
            init.flows.appendleft(cv2.calcOpticalFlowFarneback(frame0, frame, None, 0.5, 3, 4, 3, 5, 1.1, 0))
            if len(init.flows) == 10:
                Thread(target=send_request, args=(image.tobytes(), 'spatial')).start()
                # concatenate at axis 2
                # ex: (3, 2, 1) + (3, 2, 1) = (3, 2, 2)
                optical_flow = np.concatenate(init.flows, axis=2)
                Thread(target=send_request, args=(optical_flow.tobytes(), 'temporal')).start()
                init.flows.pop()

        frame0 = frame
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
        address = address['5_8k_8k51']
        for addr in address['spatial']:
            if addr == '#':
                break
            init.spatial_q.put(addr)
        for addr in address['temporal']:
            if addr == '#':
                break
            init.temporal_q.put(addr)

    server = ThreadedHTTPServer(('0.0.0.0', 9999), Handler)
    server.allow_reuse_address = True
    Thread(target=server.serve_forever, args=()).start()

    master()


if __name__ == '__main__':
    main()
