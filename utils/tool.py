import yaml
import numpy as np
import multiprocessing.managers



def read_seed():
    with open("./Configs/seed.txt",'r') as filin:
        for line in filin:
            seed=int(line)

    return seed

def write_seed(seed):
    with open("./Configs/seed.txt",'w') as fileout:
        fileout.write(str(seed))

           


def read_config(config_path):
	with open(config_path, "r") as ymlfile:
		cfg = yaml.load(ymlfile, yaml.FullLoader)

	return cfg

def rgb_to_gray(img):
	grayImage = np.zeros(img.shape)
	R = np.array(img[:, :, 0])
	G = np.array(img[:, :, 1])
	B = np.array(img[:, :, 2])

	R = (R *.299)
	G = (G *.587)
	B = (B *.114)

	Avg = (R+G+B)
	grayImage = img

	for i in range(3):
		grayImage[:,:,i] = Avg

	return grayImage


def AutoProxy(token, serializer, manager=None, authkey=None,
              exposed=None, incref=True, manager_owned=False):
    '''
    Return an auto-proxy for `token`
    '''
    _Client = multiprocessing.managers.listener_client[serializer][1]

    if exposed is None:
        conn = _Client(token.address, authkey=authkey)
        try:
            exposed = dispatch(conn, None, 'get_methods', (token,))
        finally:
            conn.close()

    if authkey is None and manager is not None:
        authkey = manager._authkey
    if authkey is None:
        authkey = multiprocessing.process.current_process().authkey

    ProxyType = multiprocessing.managers.MakeProxyType('AutoProxy[%s]' % token.typeid, exposed)
    proxy = ProxyType(token, serializer, manager=manager, authkey=authkey,
                      incref=incref, manager_owned=manager_owned)
    proxy._isauto = True
    return proxy