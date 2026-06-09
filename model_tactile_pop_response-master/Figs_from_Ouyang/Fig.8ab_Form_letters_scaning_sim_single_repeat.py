# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:11:16 2017
@author: Ouyang qiangqiang
"""
import Receptors as rslib
import numpy as np
import matplotlib.pyplot as plt
import simset as mysim
from PIL import Image
import img_to_eqstimuli as imeqst

width=120#mm
height=12#mm

shift=0.2 #mm
speed=20
pf=0.35#pressing force (N)
simT=width/speed
simdt=0.001
doth=1

Ttype_buf=['SA1','RA1','PC']
tsensors=[]
pbuf=np.load('Data/loc_pos_buf_fingertip.npy', allow_pickle=True)    
for tp in range(len(Ttype_buf)):
    tsensor=rslib.tactile_receptors(Ttype=Ttype_buf[tp])
    tsensor.set_population(pbuf[tp][0],pbuf[tp][1],simTime=simT,sample_rate=1/simdt,Density=pbuf[tp][2],roi=mysim.fingertiproi)
    tsensors.append(tsensor)
    print('{}: Wc={:.2f} mm, Hc={:.2f} mm, contact window extends +/-{:.2f} mm around probe centroid'.format(
        Ttype_buf[tp], tsensor.Wc, tsensor.Hc, tsensor.Wc/2))

img1 =Image.open('saved_figs/letters_120-12.jpg')


'Delete the "..." above and below if want to run the simulation again'
buf1, eq_stimuli, eeps, eps=imeqst.constructing_equivalent_probe_stimuli_from_image(img1,width,height,mysim.fingertiproi)
np.save('Data/forms_letters.npy',np.array([buf1, eq_stimuli, eeps, eps], dtype=object))
PF=pf*np.ones(int(simT/simdt))
ips=imeqst.img_stimuli_scaning_with_uniformal_speed(simdt,simT,PF,speed,0,width,0,height,shift)


simulation_res=[]
Aeeps=np.load('Data/forms_letters.npy', allow_pickle=True)[2]
Aeeps[1]=Aeeps[1]*doth # height of embossed letter is 1 mm according to ref. [35]

# --- inspect the EEPS that will be fed into the simulation ---
pimageinf, eepsimg = Aeeps[0], Aeeps[1]

# --------------------------------------------------------------

for tp in range(len(Ttype_buf)):
    tmp=[]
    for row in range(len(ips)):#(len(stimuli_buf1)):
        tsensors[tp].population_simulate(EEQS=Aeeps,Ips=[ips[row],'Pressure'],noise=0)
        sel=tsensors[tp].points_mapping_entrys(np.array([[0,0]]))[0]
        tmp.append(np.array(tsensors[tp].Va[sel,:]))
    simulation_res.append(tmp)
np.save('Data/letters_simulation_res_single_repeat.npy',simulation_res)
#----------------------


def diagnose_sa1_response():
    """Compare new vs original EEPS and SA1 traces to locate what changed."""
    new_eeps  = np.load('Data/forms_letters.npy',     allow_pickle=True)[2]
    orig_eeps = np.load('Data/forms_letters_org.npy', allow_pickle=True)[2]
    new_img, orig_img = new_eeps[1], orig_eeps[1]

    print('\n=== EEPS comparison ===')
    print('Original EEPS shape:', orig_img.shape, 'range:', orig_img.min(), 'to', orig_img.max())
    print('New EEPS      shape:', new_img.shape,  'range:', new_img.min(),  'to', new_img.max())
    if orig_img.shape == new_img.shape:
        diff = new_img - orig_img
        print('Pixel diff: mean={:.4e}, max|diff|={:.4e}, nonzero diff pixels={}/{}'.format(
            diff.mean(), np.abs(diff).max(), np.count_nonzero(diff), diff.size))

    fig, axes = plt.subplots(3, 1, figsize=(12, 7))
    axes[0].imshow(orig_img, cmap=plt.cm.Greys, aspect='auto'); axes[0].set_title('Original EEPS')
    axes[1].imshow(new_img,  cmap=plt.cm.Greys, aspect='auto'); axes[1].set_title('New EEPS')
    if orig_img.shape == new_img.shape:
        axes[2].imshow(new_img - orig_img, cmap=plt.cm.RdBu, aspect='auto', vmin=-1, vmax=1)
        axes[2].set_title('new - orig  (red = new only, blue = original only)')
    plt.tight_layout()
    plt.savefig('saved_figs/eeps_compare.png', bbox_inches='tight', dpi=150)
    plt.show()

    # Re-run a single SA1 scan row (mid-height) to capture internal traces
    target_row = len(ips) // 2
    py = ips[target_row][0, 1]
    print('\n=== Re-running SA1 scan row {} (y = {:.2f} mm) ==='.format(target_row, py))

    Aeeps_new = np.load('Data/forms_letters.npy', allow_pickle=True)[2]
    Aeeps_new[1] = Aeeps_new[1] * doth
    sa1 = tsensors[0]
    sa1.population_simulate(EEQS=Aeeps_new, Ips=[ips[target_row], 'Pressure'], noise=0)
    sel = sa1.points_mapping_entrys(np.array([[0, 0]]))[0]

    t       = sa1.t
    x_mm    = t * speed
    Dt_tr   = np.asarray(sa1.Dt[sel, :]).flatten()
    Uc_tr   = np.asarray(sa1.Uc[sel, :]).flatten()
    V1_tr   = np.asarray(sa1.V1[sel, :]).flatten()
    V2_tr   = np.asarray(sa1.V2[sel, :]).flatten()
    Vg_tr   = np.asarray(sa1.Vg[sel, :]).flatten()
    Va_tr   = np.asarray(sa1.Va[sel, :]).flatten()

    orig_sres = np.load('Data/letters_simulation_res_single_repeat_org.npy', allow_pickle=True)
    orig_Va = np.asarray(orig_sres[0][target_row]).flatten() if target_row < len(orig_sres[0]) else None

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    axes[0].plot(x_mm, Dt_tr, 'k', lw=0.7);     axes[0].set_ylabel('Dt (indent)')
    axes[0].set_title('SA1 internal signals at central receptor, scan row y = {:.2f} mm'.format(py))
    axes[1].plot(x_mm, Uc_tr, 'k', lw=0.7);     axes[1].set_ylabel('Uc (post R-net)')
    axes[2].plot(x_mm, V1_tr, 'b', lw=0.7);     axes[2].set_ylabel('V1 (bandpass)')
    axes[2].axhline(0, color='gray', lw=0.3)
    axes[3].plot(x_mm, V2_tr, 'm', lw=0.7);     axes[3].set_ylabel('V2 (sustained)')
    axes[4].plot(x_mm, Vg_tr, 'darkgreen', lw=0.7); axes[4].set_ylabel('Vg (rectified)')
    axes[5].plot(x_mm, Va_tr, 'g', lw=0.6, label='new')
    if orig_Va is not None and len(orig_Va) == len(Va_tr):
        axes[5].plot(x_mm, orig_Va, 'k', lw=0.4, alpha=0.6, label='original')
    axes[5].axhline(-0.05, color='r', lw=0.3, ls='--', label='spike threshold')
    axes[5].set_ylabel('Va (membrane V)'); axes[5].set_xlabel('Probe x-position [mm]')
    axes[5].legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig('saved_figs/sa1_internal_signals.png', bbox_inches='tight', dpi=150)
    plt.show()



def print_ob_letter_spiking_trians():
    plt.figure(figsize=(8,1*0.8))
    ax=plt.subplot(1,1,1)
    plt.text(-0,-30,"(a)",fontsize=14)#
    labelsx=np.round(np.linspace(0,width,7))
    labelsy=np.round(np.linspace(height,0,7))
    obimg=np.array(Image.open('saved_figs/ob_letters_80-10.jpg').convert('L'))
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    ax.imshow(obimg,cmap=plt.cm.gray,aspect='equal')
    plt.xlabel('Postion [mm]',fontsize=8) 
    plt.ylabel('Distance [mm]',fontsize=8) 
    plt.xticks(np.linspace(0,obimg.shape[1],7),labelsx,fontsize=6)
    plt.yticks(np.linspace(0,obimg.shape[0],7),labelsy,fontsize=6)
    plt.savefig('saved_figs/ob_letter_spking.png',bbox_inches='tight', dpi=300)


def print_letter_spiking_trians():
    ftiproi=np.loadtxt('Data/txtdata/fingertip_roi.txt')
    ftiproi=np.vstack([ftiproi,ftiproi[0,:]])
    plt.figure(figsize=(6,4*0.7))
    plt.subplots_adjust(hspace=0.2)
    
    buf=np.load('Data/forms_letters.npy', allow_pickle=True)[1]
    sres=np.load('Data/letters_simulation_res_single_repeat.npy', allow_pickle=True) 

    ax=plt.subplot(4,1,1)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.text(-0,14,"(b)",fontsize=14)#
    #plt.scatter(buf[:,0],buf[:,1],c=mysim.colors[3],s=0.2)
    ax.scatter(buf[:,0],buf[:,1],s=0.01,c=1e3*buf[:,5]*doth,cmap=plt.cm.Greys,vmin=0,vmax=1)
    plt.xticks(np.arange(0,width+width/10,width/10),fontsize=6)
    plt.yticks([0,height/2,height],fontsize=7)
    ax1=ax.twinx() 
    ax1.spines['top'].set_color('None')
    ax1.spines['right'].set_color('None')
    plt.yticks([])
    plt.ylabel('EPS',fontsize=8)
    num=int(height/shift)
    sel_points=np.vstack([0*np.ones(num),np.linspace(height,0,num)]).T #ms
    for ch in range(3):
        ax1=plt.subplot(4,1,ch+2,sharex=ax)
        ax1.spines['top'].set_color('None')
        ax1.spines['right'].set_color('None')
        for i in range(num):
            #res=np.array(tbuf[i][:])
            res=simdt*np.where(sres[ch][i]==0.04)[0]
            plt.scatter(res*speed,sel_points[i,1]*np.ones(len(res)),c=mysim.colors[ch],marker='.',s=0.01)
            plt.xticks(np.arange(0,width+width/10,width/10),fontsize=7)
            plt.yticks([0,4,8,12],fontsize=7)
        if(ch==2):plt.xlabel('Postion [mm]',fontsize=10) 
        if(ch==1):plt.ylabel('     Distance [mm]',fontsize=10)   
        ax2=ax1.twinx() 
        ax2.spines['top'].set_color('None')
        ax2.spines['right'].set_color('None')
        plt.yticks([])
        plt.ylabel(Ttype_buf[ch],fontsize=8)
    plt.savefig('saved_figs/letter_spking.png',bbox_inches='tight', dpi=300)     
  
       
print_ob_letter_spiking_trians()
print_letter_spiking_trians()

img1=Image.open('saved_figs/ob_letter_spking.png')
img2=Image.open('saved_figs/letter_spking.png')

img=Image.new(img1.mode,(img2.size[0],60+img1.size[1]+img2.size[1]))

img.paste(img1,(60,0))
img.paste(img2,(-10,img1.size[1]))
img.save("saved_figs/submitting/Form_letters_all.png")
img.show()
