"""
# -*- coding: utf-8 -*-
Created on Sat Nov 18 17:11:16 2017
@author: qiangqiang ouyang
"""
'''
https://www.cnblogs.com/smallpi/p/4555854.html
'''
import os 
import sys
import utils as alt
import Receptors as rslib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import simset as mysim 
from skimage import transform
import matplotlib.cm as cm
import scipy.signal as signal


# a function that generates a Gaussian operator
def func(x,y,sigma=1):
    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))

def rgb2gray(rgb):
    """
    Convert an RGB image array to greyscale using standard luminosity weights.
 
    Weights (0.2989 R, 0.5870 G, 0.1140 B) approximate human colour perception.
 
    Args:
        rgb : H*W*3
    Returns:
        H*W numpy array of float greyscale values.
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

    
#image.reshape(int(h/mysim.prope_d),int(w/mysim.prope_d))
#image=transform.resize(image,(int(h/mysim.prope_d),int(w/mysim.prope_d))) #比例缩放
def constructing_equivalent_probe_stimuli_from_pimage(pimage,w,h,roi):
    'probe image'
    pimageinf=np.array([w,h,pimage.shape[1],pimage.shape[0]])
    
    bw=roi[:,0].max()-roi[:,0].min()
    bh=roi[:,1].max()-roi[:,1].min()
    cols=int(bw/rslib.Dbp)
    rows=int(bh/rslib.Dbp)
    '''
    Or=[int(0-roi[:,0].min()/bw*cols),
        int(0-roi[:,1].min()/bh*rows)]  
    #extend EQS
    EPS=np.array(pimage)
    A=np.array(pimage)
    A=np.hstack([np.zeros([A.shape[0],Or[0]]),A])
    A=np.hstack([A,np.zeros([A.shape[0],cols-Or[0]])])
    A=np.vstack([A,np.zeros([rows-Or[1],A.shape[1]])])
    EEPS=np.vstack([np.zeros([Or[1],A.shape[1]]),A])
    '''
    #extend EQS
    EPS=np.array(pimage)
    A=np.array(pimage)
    A=np.hstack([np.zeros([A.shape[0],cols]),A])
    A=np.hstack([A,np.zeros([A.shape[0],cols])])
    A=np.vstack([A,np.zeros([rows,A.shape[1]])])
    EEPS=np.vstack([np.zeros([rows,A.shape[1]]),A])
    
    'equvilent stimuli dots'
    selimg=pimage
    dots=np.meshgrid(np.linspace(0,w,selimg.shape[1]),np.linspace(h,0,selimg.shape[0]))
    x=dots[0].reshape(dots[0].size,1)
    y=h-dots[1].reshape(dots[1].size,1)
    th=selimg.reshape(selimg.size,1)
    tmp=np.hstack([x,y,th])
    pins=tmp[:,:]
    x,y,th=pins[:,0:1],pins[:,1:2],pins[:,2:3]
    eq_stimuli=np.hstack([x,y,x*0,x*0,np.ones([x.size,1])*rslib.Dbp*1e-3,th*1e-3])


    return [pimageinf,EEPS,eq_stimuli,EPS]


"""
Convert a standard RGB image into an EPA and EEPA for tactile simulation.

Full processing pipeline:
    1. Greyscale + normalise
    2. Edge detection (Gaussian smooth -> Laplace)
    3. Resize to probe resolution (Dbp spacing)
    4. Binarise probe heights
    5. Zero-pad to create the Extended EPA (EEPA)
    6. Output probe position/height matrix for the biomechanical model

Args:
    img : PIL Image or HxWx3 numpy array (RGB)
    w   : physical width of the stimulus surface in mm
    h   : physical height of the stimulus surface in mm
    roi : Nx2 array of (x, y) coordinates defining the fingertip ROI

Returns:
    res_buf     : list of intermediate images (see index table below)
    eq_stimuli  : Mx6 array of active probe positions and heights
    [pimageinf, eepsimg] : EEPA metadata + image
    [pimageinf, epsimg]  : EPA  metadata + image

res_buf index table:
    [0]  [w, h]                        physical dimensions
    [1]  original image array
    [2]  greyscale normalised image
    [3]  edge-detected image
    [4]  height image with edge enhancement
    [5]  EPA image (probe-resolution binary)
    [6]  EEPA image (zero-padded EPA)
    [7]  active probe pin positions [x, y, height]
"""
def constructing_equivalent_probe_stimuli_from_image(img,w,h,roi):

    # Generate a 5*5 Gaussian operator with a standard deviation of 5
    # 15x15 Gaussian kernel (sigma=3) for smoothing before edge detection
    suanzi1 = np.fromfunction(func,(15,15),sigma=3)
    # Laplace extent operator
    suanzi2 = np.array([[1, 1, 1],
                        [1,-8, 1],
                        [1, 1, 1]])
    # ── Step 0: store physical dimensions ─────────────────────────────────
    'Original image'
    res_buf=[]
    res_buf.append([w,h])
    # ── Step 1: store original image ──────────────────────────────────────
    res_buf.append(np.array(img))
    
    'Grayscale image'
    # ── Step 2: greyscale conversion + normalisation ───────────────────────
    # Convert RGB -> greyscale, then scale to [0, 1]
    grayimg=np.array(rgb2gray(np.array(img)))
    grayimg =grayimg * (1 / 255)  #归一化
    #grayimg=grayimg/50 # color height
    res_buf.append(grayimg)
  
    'edge dection'
    # ── Step 3: edge detection ─────────────────────────────────────────────
    image2=np.array(grayimg)
    
    image2[0:2,:]=1
    image2[image2.shape[0]-1:image2.shape[0],:]=1
    image2[:,0:1]=1
    image2[:,image2.shape[1]-1:image2.shape[1]]=1
    #grayimg =(grayimg - grayimg.min()) * (1 / (grayimg.max() - grayimg.min()))  #归一化
    
    # Using the generated Gaussian operator to convolve with the original image to smooth the image
    image2 = signal.convolve2d(image2, suanzi1, mode="same")
    
    # edge dection to image
    image2 = signal.convolve2d(image2, suanzi2, mode="same")
    # normalize the pixel value of image int 0-1
    image2 = (image2 - image2.min()) * (1 / (image2.max() - image2.min())) 
    #(image2/float(image2.max()))*1
    #Make the gray value larger than the gray average value 0 (white) for easy observation of the edge
    image2[image2>image2.mean()] = 0
    
    #fill zeros to pixels beside edge of image
    num=9
    image2[0:num,:]=0
    image2[image2.shape[0]-num:image2.shape[0],:]=0
    image2[:,0:num]=0
    image2[:,image2.shape[1]-num:image2.shape[1]]=0
    
    edgimage=image2
    res_buf.append(np.array(edgimage))
    
        # ── Step 4: height image with edge enhancement ─────────────────────────
    # Normalise edge image and combine with greyscale to create height map
    'obtian height image with edge enhancement'
    edimage=(edgimage - edgimage.min()) * (1 / (edgimage.max() - edgimage.min()))
    imageh=grayimg#+edimage
    
    imageh=(imageh - imageh.min()) * (1 / (imageh.max() - imageh.min()))
    res_buf.append(imageh)

    # ── Step 5: resize to EPA probe resolution ─────────────────────────────
    # Downsample so each pixel corresponds to one probe (spacing = Dbp mm)
    # EPA dimensions: rows = h/Dbp, cols = w/Dbp  (Eq. 1 in the paper)    
    'EPS image'
    pimage=transform.resize(imageh,(int(h/rslib.Dbp),int(w/rslib.Dbp)))
    pimage[pimage>=0.1]=1
    pimage[pimage<0.1]=0
    
    epsimg=pimage
    res_buf.append(pimage)
    pimageinf=np.array([w,h,pimage.shape[1],pimage.shape[0]])
    

    # ── Step 6: zero-pad to create the EEPA ───────────────────────────────
    # The EEPA extends the EPA by the fingertip contact area dimensions so
    # every probe position can be scanned (Section A.1 of the paper).
    bw=roi[:,0].max()-roi[:,0].min()
    bh=roi[:,1].max()-roi[:,1].min()
    cols=int(bw/rslib.Dbp)
    rows=int(bh/rslib.Dbp)
    
    '''
    Or=[int(0-roi[:,0].min()/bw*cols),
        int(0-roi[:,1].min()/bh*rows)]  
    #extend EQS
    'extended EQS '
    A=np.array(pimage)
    A=np.hstack([np.zeros([A.shape[0],Or[0]]),A])
    A=np.hstack([A,np.zeros([A.shape[0],cols-Or[0]])])
    A=np.vstack([A,np.zeros([rows-Or[1],A.shape[1]])])
    A=np.vstack([np.zeros([Or[1],A.shape[1]]),A])
    '''
    A=np.array(pimage)
    A=np.hstack([np.zeros([A.shape[0],cols]),A])
    A=np.hstack([A,np.zeros([A.shape[0],cols])])
    A=np.vstack([A,np.zeros([rows,A.shape[1]])])
    A=np.vstack([np.zeros([rows,A.shape[1]]),A])
    
    
    eepsimg=A
    res_buf.append(eepsimg)
    
    
    # ── Step 7: generate active probe positions ────────────────────────────
    # Map each EPA pixel to physical (x, y) coordinates in mm
    'equvilent stimuli dots'
    selimg=pimage
    dots=np.meshgrid(np.linspace(0,w,selimg.shape[1]),np.linspace(h,0,selimg.shape[0]))
    x=dots[0].reshape(dots[0].size,1)
    y=dots[1].reshape(dots[1].size,1)
    th=selimg.reshape(selimg.size,1)
    tmp=np.hstack([x,y,th])
    pins=tmp[tmp[:,2]>0.1,:]
    res_buf.append(pins)
    
    x,y,th=pins[:,0:1],pins[:,1:2],pins[:,2:3]
    # Build 6-column stimulus matrix for the biomechanical skin model:
    # [x_mm, y_mm, 0, 0, probe_diameter_m, probe_height_m]
    eq_stimuli=np.hstack([x,y,x*0,x*0,np.ones([x.size,1])*mysim.prope_d*1e-3,th*1e-3])
    #return EEQS
    # 显示图像
    '''
    plt.figure(figsize=(6,3*2))
    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(3,1,2)
    plt.imshow(pimage,cmap=cm.Greys)
    plt.axis("off")
    
    plt.subplot(3,1,3)
    plt.scatter(pins[:,0],pins[:,1],s=0.5,c=pins[:,2],cmap=cm.Greys,vmin=0,vmax=3)

    plt.show()
    '''
    return res_buf,eq_stimuli,[pimageinf,eepsimg],[pimageinf,epsimg]


"""
Generate a sequence of static-press stimuli by stepping the contact point
horizontally across the stimulus surface.

At each step the finger presses at a fixed (x, y) location for duration T,
then shifts right by `shift` mm until the right edge is reached.

Args:
    dt    : time step in seconds
    T     : press duration in seconds
    pf    : pressure-force waveform array, length = int(T/dt)
    buf   : unused in this function (reserved for future use)
    roi   : fingertip ROI (unused here, reserved)
    spx   : starting x position in mm
    spy   : y position in mm (fixed throughout)
    shift : horizontal step size in mm between presses
    w     : total stimulus width in mm (defines stopping condition)

Returns:
    stimuli_buf : list of arrays, each of shape (int(T/dt), 3)
                    columns = [x_mm, y_mm, force]
"""
def img_stimuli_static_pressing(dt,T,pf,buf,roi,spx,spy,shift,w):
    slide=0
    stimuli_buf=[]
    while(1):   
        ips=np.zeros([int(T/dt),3])
        for i in range(int(T/dt)):
            [ox,oy]=[spx+slide,spy]
            ips[i,:]=[ox,oy,pf[i]]
        slide=slide+shift
        stimuli_buf.append(ips)
        if(slide+spx>w):break
    return stimuli_buf


"""
Generate a scanning stimulus where the contact point moves at constant
speed along the scan direction, then shifts perpendicular between sweeps.

Models a finger sliding across the surface row by row (like a raster scan).

Args:
    dt           : time step in seconds
    T            : total scan duration per sweep in seconds
    pf           : pressure-force waveform, length = int(T/dt)
    speed        : scanning speed in mm/s
    sp_scandir   : start position in the scan direction (mm)
    end_scandir  : end position in the scan direction (mm)
    sp_shiftdir  : start position in the shift direction (mm)
    end_shiftdir : end position in the shift direction (mm)
    shift        : step size in the shift direction (mm) between sweeps

Returns:
    stimuli_buf : list of arrays, each of shape (int(T/dt), 3)
                columns = [x_mm, y_mm, force]
"""
def img_stimuli_scaning_with_uniformal_speed(dt,T,pf,speed,sp_scandir,end_scandir,sp_shiftdir,end_shiftdir,shift):
    vslide=0
    stimuli_buf=[]
    while(1): 
        ips=np.zeros([int(T/dt),3])
        for i in range(int(T/dt)):
            [ox,oy]=[sp_scandir+i*dt*speed,sp_shiftdir+vslide]
            ips[i,:]=[ox,oy,pf[i]]
            if(ox>end_scandir):break
        vslide=vslide+shift
        stimuli_buf.append(ips)
        if(vslide>end_shiftdir):break
    return stimuli_buf


"""
Plot the full image-processing pipeline for one stimulus image into a
shared matplotlib figure.

Designed to be called twice (s=-1 and s=0) to display two images side
by side (e.g. artificial vs natural texture) in an 8-row x 2-column grid.

Args:
    s   : column offset - use -1 for the left column, 0 for the right
    buf : list returned by constructing_equivalent_probe_stimuli_from_image()
            index 0 = [w, h], 1 = original, 2 = greyscale, 3 = edge,
            4 = height, 5 = EPS, 6 = EEPS
    fig : matplotlib Figure object to draw into
"""
def print_figs(s,buf,fig):
    # Use linspace + round to avoid float-precision artifacts (e.g. 7.1999999
    # instead of 7.2 when HEIGHT_MM=12). np.arange with float steps accumulates
    # error; linspace is exact at the endpoints.
    labelsy=np.round(np.linspace(0,buf[0][1],6),1)
    labelsx=np.uint16(np.round(np.linspace(0,buf[0][0],6)))
    
    ax = fig.add_subplot(8,2,2+s)
    plt.title('Original  image')
    img=buf[1]
    ax.imshow(img,aspect='auto')
    plt.xticks(np.linspace(0,img.shape[1],6),labelsx,fontsize=8)
    plt.yticks(np.linspace(0,img.shape[0],6),labelsy,fontsize=8)
    
    ax = fig.add_subplot(8,2,4+s,sharex=ax)
    plt.title('Grayscale image after normalization')
    img=buf[2]
    md=ax.imshow(img,cmap=cm.Greys,aspect='auto')
    plt.xticks(np.linspace(0,img.shape[1],6),labelsx,fontsize=8)
    plt.yticks(np.linspace(0,img.shape[0],6),labelsy,fontsize=8)
    if(s==-1):plt.colorbar(md,cax=fig.add_axes([0.48, 0.33, 0.01, 0.25]))
    '''
    ax = fig.add_subplot(6,2,6+s,sharex=ax)
    plt.title('Edge dection using Laplace operator')
    img=buf[3]
    ax.imshow(img,cmap=cm.Greys,aspect='auto')
    plt.xticks(np.linspace(0,img.shape[1],6),labelsx,fontsize=8)
    plt.yticks(np.linspace(0,img.shape[0],6),labelsy,fontsize=8)
    '''
    ax = fig.add_subplot(8,2,6+s,sharex=ax)
    plt.title('Height image with edge enhancement')
    img=buf[4]
    ax.imshow(img,cmap=cm.Greys,aspect='auto')
    plt.xticks(np.linspace(0,img.shape[1],6),labelsx,fontsize=8)
    plt.yticks(np.linspace(0,img.shape[0],6),labelsy,fontsize=8)
   
    ax = fig.add_subplot(8,2,8+s)
    plt.title('Equivalent probe stimuli (EPS) image')
    img=buf[5]
    ax.imshow(img,cmap=cm.Greys,aspect='auto',vmin=0,vmax=1)
    plt.xticks(np.linspace(0,img.shape[1],6),labelsx,fontsize=8)
    plt.yticks(np.linspace(0,img.shape[0],6),labelsy,fontsize=8)
    
    ax = fig.add_subplot(4,2,6+s)
    

    plt.title('Extended EPS (EEPS) image')
    img=buf[6]
    ax.imshow(img,cmap=cm.Greys,aspect='auto')
    ftiproi=np.loadtxt('Data/txtdata/fingertip_roi.txt')
    ftiproi=img.shape[0]-np.vstack([ftiproi,ftiproi[0,:]])*9
    plt.plot(ftiproi[:,0]-ftiproi[:,0].min(),ftiproi[:,1]-ftiproi[:,1].min(),'y-',linewidth=1)
    
    plt.fill_between(ftiproi[:,0]-ftiproi[:,0].min(),ftiproi[:,1]-ftiproi[:,1].min(),facecolor='y',alpha=0.3) 
    plt.scatter([25],[50],s=60,c='k',marker='+')
    plt.xticks(np.linspace(0,img.shape[1],6),np.round(1.5*labelsx,1),fontsize=8)
    plt.yticks(np.linspace(0,img.shape[0],6),np.round(3.2*labelsy,1),fontsize=8)
    plt.xlabel('x [mm]',fontsize=8)
    plt.ylabel('y [mm]',fontsize=8)
   
    '''
    ax = fig.add_subplot(6,2,10+s)
    plt.title('Equivalent probe stimuli')
    pins=buf[7]
    ax.scatter(pins[:,0],pins[:,1],s=0.5,c=pins[:,2],cmap=cm.Greys,vmin=0,vmax=3)
    plt.xticks(labelsx,fontsize=8)
    plt.yticks(labelsy,fontsize=8)
    '''


# ─────────────────────────────────────────────────────────────────────────────
# Example usage (uncomment to run)
# ─────────────────────────────────────────────────────────────────────────────
'''
nat_image =Image.open('saved_figs/letters_120-12.jpg')
art_image =Image.open('saved_figs/letters_ISA.jpg')
art_buf=constructing_equivalent_probe_stimuli_from_image(art_image,30,10,mysim.fingertiproi)[0]
nat_buf=constructing_equivalent_probe_stimuli_from_image(nat_image,30,10,mysim.fingertiproi)[0]
fig =plt.figure(figsize=(10,9*2)) 
plt.subplots_adjust(hspace=0.5) 
plt.subplots_adjust(wspace=0.4) 
print_figs(-1,art_buf,fig)
print_figs(0,nat_buf,fig)
plt.savefig('saved_figs/image_to_stimuli.png', bbox_inches='tight',dpi=300) 
'''