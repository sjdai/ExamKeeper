import socket
import threading
import struct
import binascii

#socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#bind
pars = ('140.109.22.72', 5181)
s.bind(pars)

#listen
s.listen(5)

#recv send close
def packet(id,seq,msg):
    Type = 0
    Code = 0
    Unused = 65535
    Identifier = id
    SequenceNumber = seq
    Message = msg
    return struct.pack('!BBHHH3sx',Type,Code,Unused,Identifier,SequenceNumber,Message)

def print_packet(pkt):
    print(binascii.hexlify(pkt))
    Type,Code,Unused,Identifier,SequenceNumber,Message = struct.unpack('!BBHHH3sx',pkt)
    print("Type : {}".format(Type))
    print("Code : {}".format(Code))
    print("Unused : {}".format(Unused))
    print("Identifier : {}".format(Identifier))
    print("SequenceNumber : {}".format(SequenceNumber))
    print("Message : {}".format(Message))
    print()

def serveClient(clientsocket, address):
    while True:
        data = clientsocket.recv(12)
        if data:
            Type,Code,Unused,Identifier,SequenceNumber,Message = struct.unpack('!BBHHH3sx',data)
            print('Received Packet:')
            print_packet(data)

            sending_packet = packet(Identifier,SequenceNumber,Message)
            clientsocket.send(sending_packet)
            print('Sending_packet')
            print_packet(sending_packet)

        if Message == 'cls':
            clientsocket.close()
            break
#accept
while True:
    (clientsocket, address) = s.accept()
    threading.Thread(target = serveClient, args = (clientsocket, address)).start()
