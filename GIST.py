import numpy as np
import numpy.matlib as nm
import numpy.fft as f
from PIL import Image

def _createGabor(orr,n):

    gabor_param = []
    Nscalse = len(orr)
    Nfilters = sum(orr)

    if len(n) == 1:
        n = [n[0],[0]]
    for i in range(Nscalse):
        for j in range(orr[i]):
            gabor_param.append([.35,.3/(1.85**(i)),16*orr[i]**2/32**2, np.pi/(orr[i])*(j)])
    gabor_param = np.array(gabor_param)

    fx, fy = np.meshgrid(np.arange(-n[1]/2,n[1]/2-1 + 1), np.arange(-n[0]/2, n[0]/2-1 + 1))
    fr = f.fftshift(np.sqrt(fx**2+fy**2))
    t = f.fftshift(np.angle(fx+ 1j*fy))

    G = np.zeros([n[0],n[1],Nfilters])
    for i in range(Nfilters):
        tr = t + gabor_param[i,3]
        tr+= 2*np.pi*(tr < -np.pi) - 2 * np.pi*(tr>np.pi)
        a = np.exp(-10*gabor_param[i,0]*(fr/n[1]/gabor_param[i,1]-1)**2-2*gabor_param[i,2]*np.pi*tr**2)
        G[:,:,i] = np.exp(-10*gabor_param[i,0]*(fr/n[1]/gabor_param[i,1]-1)**2-2*gabor_param[i,2]*np.pi*tr**2)

    return G

def _more_config(img,param):

    param["imageSize"] = [img.shape[0], img.shape[1]]
    param["G"] = _createGabor(param["orientationsPerScale"],np.array(param["imageSize"])+2*param["boundaryExtension"])
    return param

def _preprocess(img, M):
    if len(M) == 1:
        M = [M, M]
    scale = np.max([M[0]/img.shape[0], M[1]/img.shape[1]])


    newsize = np.round(np.array([img.shape[0],img.shape[1]]) * scale)
    img = np.array(Image.fromarray(img).resize(newsize, Image.BILINEAR))
    #img = imresize(img,newsize,'bilinear')
    nr,nc = img.shape
    sr = np.floor((nr-M[0])/2)
    sc = np.floor((nc-M[1])/2)
    img = img[int(sr):int(sr+M[0]) + 1,int(sc):int(sc+M[1])+1]
    img = img- np.min(img[:])
    img = 255*(img/np.max(img[:]))
    return img



def _prefilt(img, fc):
    
    w = 5
    s1 = fc/np.sqrt(np.log(2))
    img=np.log(img +1 )
    img = np.pad(img,[w,w],"symmetric")

    sn,sm = img.shape
    n = np.max([sn,sm])
    n += n%2

    img = np.pad(img,[0,int(n-sm)],"symmetric")

    fx,fy = np.meshgrid(np.arange(-n/2,n/2-1 + 1),np.arange(-n/2,n/2-1 + 1))
    gf = f.fftshift((np.exp(-(fx**2+fy**2)/(s1**2))))
    gf = nm.repmat(gf,1,1)
    output = img - np.real(f.ifft2(f.fft2(img)*gf))

    localstd = nm.repmat(np.sqrt(abs(f.ifft2(f.fft2(output**2)*gf))), 1 ,1 )
    output = output/(0.2+localstd)
    output = output[w:sn-w, w:sm-w]
    return output

def _gistGabor(img,param):

    w = param["numberBlocks"]
    G = param["G"]
    be = param["boundaryExtension"]
    ny,nx,Nfilters = G.shape
    W = w * w
    N = 1
    g = np.zeros((W*Nfilters, N))
    img = np.pad(img,[be,be],"symmetric")
    img = f.fft2(img)
    
    k = 0
    for n in range(Nfilters):
        ig = abs(f.ifft2(img*nm.repmat(G[:,:,n],1,1)))
        ig = ig[be:ny-be,be:nx-be]
        v = _downN(ig,w)
        g[k:k+W,0] = v.reshape([W,N],order = "F").flatten()
        k += W
    return np.array(g)
    
def _downN(x,N):
    nx = list(map(int,np.floor(np.linspace(0,x.shape[0],N+1))))
    ny = list(map(int,np.floor(np.linspace(0,x.shape[1],N+1))))
    y  = np.zeros((N,N))
    for xx in range(N):
        for yy in range(N):
            a = x[nx[xx]:nx[xx+1], ny[yy]:ny[yy+1]]
            v = np.mean(np.mean(a,0))
            y[xx,yy]=v
    return y

def _gist_extract(img, param):

    param = _more_config(img,param)

    img = _preprocess(img, param["imageSize"])

    output = _prefilt(img,param["fc_prefilt"])

    gist = _gistGabor(output,param)

    return gist

