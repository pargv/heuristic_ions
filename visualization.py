import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx 

rc = {"font.family" : "Times New Roman", 
      "mathtext.fontset" : "cm"}
plt.rcParams.update(rc)

def set_label_font(ax,fs):
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fs) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fs) 
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

def plot_expr(L, expr_sets, labels, markers, colors, ymax=1e1):

    fig = plt.figure(figsize=(6,4))
    ax = plt.gca()

    for i, (expr, label) in enumerate(zip(expr_sets, labels)):
        ax.scatter(L, expr, color=colors[i], s=75, marker=markers[i], 
                   edgecolor='black', lw=0.7, label=label)
        ax.plot(L, expr, '--', color=colors[i], lw=0.75)

    ax.set_yscale('log')
    ax.set_xlabel('$p$', fontsize=22)
    ax.set_ylabel('$D_{\mathrm{KL}}$', fontsize=22, labelpad=5)
    ax.set_ylim([1e-4, ymax])
    ax.set_xticks(L)
    ax.legend(fontsize=16,framealpha=1.0,loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.35),handleheight=1.0, 
              labelspacing=0.01)
    set_label_font(ax,fs=16)
    
    ax.grid(visible=True, which='minor', axis='both',color='tab:gray', linestyle=(0, (1, 3)), linewidth=1.0, alpha=0.3)
    ax.grid(visible=True, which='major', axis='both',color='tab:gray', linestyle='-', linewidth=1.0, alpha=0.2)

    plt.show()

            
def plot_distributions(n,n_bins,F,xmax=1,half_dim=1):

    fig = plt.figure(figsize = (6,4))
    ax = plt.gca()
    
    pdf, x = np.histogram(F,bins=n_bins,density=True,range=(0,1))
    x = (x[1:]+x[:-1])/2.0
    
    N = 2**(n - half_dim)
    f_Haar = lambda x: (N-1)*(1.0-x)**(N-2)
    pdf_Haar = f_Haar(x)
    
    ax.hist(F,bins=n_bins,density=True, range=(0,1), color='lightsteelblue', label = 'анзац', 
            alpha=1.0, edgecolor='black', lw=0.4)
    
    ax.plot(x, pdf_Haar, '--', color='tab:red', lw = 1.75, label = 'Хаар')
    
    ax.set_xlabel('$x$', fontsize = 28)
    ax.set_ylabel('$\\rho(x)$', fontsize = 28, labelpad = 5)
    if xmax < 0.1:
        ax.set_xticks(np.linspace(0.0,0.1,11))
    else:
        ax.set_xticks(np.linspace(0.0,1.0,11))
    ax.set_xlim([0,xmax])
    ax.legend(fontsize = 24)
    set_label_font(ax,fs=22)
    
    plt.show()
    
def plot_energies(data_sets, labels, markers, colors, lvl, ymax):

    fig = plt.figure(figsize = (6,4))
    ax = plt.gca()
    
    for i, (data, label) in enumerate(zip(data_sets, labels)):
        L = data[:,0]
        energy = data[:,1]
        ax.scatter(L,energy, color = colors[i], s = 75, marker = markers[i], 
                   edgecolor='black', lw=0.7, label=label)
        ax.plot(L, energy, '--', color = colors[i], lw = 0.75)
    
    ax.axhline(y=lvl, ls='--', color='tab:gray', lw=1.5)
    
    ax.set_xlabel('$p$', fontsize = 28)
    ax.set_ylabel('$E$', fontsize = 28, labelpad = 5)
    ax.set_ylim(ymax=ymax)
    ax.set_xticks(L)
    ax.legend(fontsize=20)
    set_label_font(ax,fs=22)
    
    plt.show()
    
    
def plot_energies_log(data_sets, labels, markers, colors, lvl, ymin):

    fig = plt.figure(figsize = (6,4))
    ax = plt.gca()
    
    if lvl is not None:
        ax.axhline(y=lvl, ls='--', color='tab:red', lw=1.5, zorder=0)
    
    for i, (data, label) in enumerate(zip(data_sets, labels)):
        L = data[:,0]
        energy = data[:,1]
        ax.scatter(L,energy, color = colors[i], s = 75, marker = markers[i], 
                   edgecolor='black', lw=0.9, label=label)
        ax.plot(L, energy, '--', color = colors[i], lw = 0.75)
    
    
    ax.set_xlabel('$p$', fontsize = 24)
    ax.set_ylabel('$1 - r$', fontsize = 24, labelpad = 5)
    ax.set_xticks(L)
    ax.set_yscale('log')
    ax.set_ylim(ymin=0.5e-6)
    ax.grid(visible=True, which='minor', axis='both',color='tab:gray', linestyle=(0, (1, 3)), linewidth=1.0, alpha=0.3)
    ax.grid(visible=True, which='major', axis='both',color='tab:gray', linestyle='-', linewidth=1.0, alpha=0.2)
    ax.legend(fontsize=18,framealpha=1.0,loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.3),handleheight=1.0, 
              labelspacing=0.01, handletextpad=-0.1)
    ax.set_ylim(ymin=ymin)
    set_label_font(ax,fs=18)
    
    plt.show()
    
def plot_energies_log_avg(data_sets, labels, markers, colors, ymin, ncol=2, loc=1):

    p = data_sets[0].shape[0]
    fig = plt.figure(figsize = (0.6*p,3.5))
    ax = plt.gca()
    
    for i, (data, label) in enumerate(zip(data_sets, labels)):
        L = data[:,0]
        energy = data[:,1]
        var = data[:,2]
        ax.scatter(L,energy, color = colors[i], s = 75, marker = markers[i], 
                   edgecolor='black', lw=0.9, label=label)
        ax.plot(L, energy, '--', color = colors[i], lw = 0.75)
        ax.fill_between(L,energy-2*var,energy+2*var, color = colors[i], alpha=0.2)
    
    ax.set_xlabel('$p$', fontsize = 22)
    ax.set_ylabel('$1 - r$', fontsize = 22, labelpad = 5)
    ax.set_xticks(L)
    ax.set_yscale('log')
    ax.set_ylim(ymin=0.5e-6)
    ax.grid(visible=True, which='minor', axis='both',color='tab:gray', linestyle=(0, (1, 3)), linewidth=1.0, alpha=0.3)
    ax.grid(visible=True, which='major', axis='both',color='tab:gray', linestyle='-', linewidth=1.0, alpha=0.2)
    ax.legend(loc=loc,ncol=ncol,fontsize=16,framealpha=1.0)
    ax.set_ylim(ymin=ymin)
    set_label_font(ax,fs=16)
    
    plt.show()
    
def draw_graph(G, pos, label_pos = 0.5):
    plt.figure(figsize=(5,4))
    
    labels = {}
    for i in range(len(G.nodes())):
        labels[i] = r'$'+str(i+1)+'$'
    

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx(G,node_color = 'black', node_size = 1000, alpha = 1, font_size=0, pos=pos)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, \
                                    label_pos=label_pos, font_size=12)
    nx.draw_networkx_labels(G, pos, labels, font_size=22, font_color='white')
   
    plt.show()
    
    
def plot_landscape(landscape):
    import matplotlib.cm as cm
    colormap = plt.colormaps['coolwarm']
    
    n = landscape.shape[0]
    
    fig = plt.figure(figsize = (4,4))
    ax = fig.add_subplot(1, 1, 1)
    
    i0, j0 = np.unravel_index(landscape.argmin(), landscape.shape)
    x_opt = [i0, j0]
    
    ax.scatter(x_opt[1], x_opt[0], color = 'tab:green', s = 35, marker = 'o', 
                edgecolor='black', lw=0.7)
    
    major_ticks_x = np.arange(0, n, n//4)
    major_ticks_y = np.arange(0, n, n//4)
    ax.set_axisbelow(True)
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticklabels(['$0$', '$\pi/8$', '$\pi/4$', '$3\pi/8$', '$\pi/2$'], fontsize = 16)
    ax.set_yticklabels(['$0$', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'], fontsize = 16)
    
    ax.set_xlabel('$\\beta$', fontsize = 18)
    ax.set_ylabel('$\gamma$', fontsize = 18, labelpad = -5.0)

    ax.set_frame_on(True) 

    
    sc = ax.imshow(landscape,interpolation = 'spline16', cmap = colormap, rasterized=True, origin = 'lower')
    
    cbaxes = fig.add_axes([0.92, 0.14, 0.03, 0.6]) 
    cb = plt.colorbar(sc, cax = cbaxes, orientation = 'vertical')
    cb.ax.tick_params(labelsize=14)
    
    ax.text(x=n,y=n-3,s='$E(\\beta,\gamma)$', fontsize=18)
       
    plt.show()
    
def plot_energy_hist(energies,s):
    
    fig = plt.figure(figsize=(5,3))
    ax = plt.gca()
    ax.hist(energies, bins=20,color='lightsteelblue', 
            alpha=1.0, edgecolor='black', lw=0.4)
    ax.set_ylabel('counts', fontsize=20)
    ax.set_xlabel(s, fontsize=20)

    print('av: ', np.mean(energies))
    print('min: ', np.min(energies))
    
    set_label_font(ax,16)
    
    plt.show()
    
def plot_ovlps_hist(ovlps):
    
    fig = plt.figure(figsize=(4.5,3))
    ax = plt.gca()
    ax.hist(ovlps, bins=20,color='tab:blue', 
            alpha=1.0, edgecolor='black', lw=1.0)
    ax.set_ylabel('counts', fontsize=18)
    ax.set_xlabel('overlap', fontsize=18)
    ax.set_xlim([0,1])

    print('avg ovlp: ', np.mean(ovlps))
    print('min ovlp: ', np.min(ovlps))
    
    set_label_font(ax,16)
    
    plt.show()
    
def plot_bars(success):
    
    fig = plt.figure(figsize=(8,5))
    ax = plt.gca()
    
    x = [0,1]
    counts = [np.sum(success == False)/len(success), np.sum(success)/len(success)]

    ax.bar(x=x,height=counts, width=0.5)
    ax.set_xticks([0,1])
    ax.set_yticks(np.linspace(0,1,6))
    ax.set_xlabel('success', fontsize=20)
    ax.set_ylabel('probability', fontsize=20)
    set_label_font(ax,16)
    plt.show()
    
def plot_bars_n(ns,data_sets,labels,colors):
    
    fig = plt.figure(figsize=(6,3.5))
    ax = plt.gca()
    
    nk = len(data_sets)
    
    ax.grid(visible=True, which='major', axis='both',color='tab:gray', linestyle=(0, (1, 3)), linewidth=1.0, alpha=0.5, zorder=0)
    
    for i, (data, label) in enumerate(zip(data_sets, labels)):
        
        ax.bar(x=ns,height=data, color=colors[i], width=0.7, edgecolor='black', lw=1.5, label = label, alpha=1.0,zorder=nk-i)
        
    ax.set_xticks(ns)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_xlabel('$n$', fontsize=22)
    ax.set_ylabel('Fraction of solved instances', fontsize=16)
    ax.set_ylim([0.0,1.05])
    ax.legend(loc=3,fontsize=16,framealpha=1.0)

    set_label_font(ax,16)
    plt.show()
    
def plot_frac_solved(p,data_sets,labels,markers,colors,ncol=2):
    
    fig = plt.figure(figsize=(0.6*len(p),3.5))
    ax = plt.gca()
    
    for i, (data, label) in enumerate(zip(data_sets, labels)):
        
        ax.scatter(p, data, color = colors[i], s = 55, marker = markers[i], 
                   edgecolor='black', lw=0.9, label=label)
        ax.plot(p, data, '--', color = colors[i], lw = 0.75)
        
    ax.set_xticks(p)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_xlabel('$p$', fontsize=22)
    ax.set_ylabel('Fraction of solved instances', fontsize=16)
    ax.set_ylim([0.0,1.05])
    ax.legend(loc=4,fontsize=16,framealpha=1.0,ncol=ncol,handleheight=1.0, 
              labelspacing=0.075,columnspacing=0.025,handletextpad=-0.1)
    ax.grid(visible=True, which='major', axis='both',color='tab:gray', linestyle=(0, (1, 3)), linewidth=1.0, alpha=0.5)
    set_label_font(ax,16)
    plt.show()
    
def plot_avg_ovlp(p,data_sets,labels,markers,colors,ncol=2,loc=2):
    
    fig = plt.figure(figsize=(0.6*len(p),3.5))
    ax = plt.gca()
    
    for i, (data, label) in enumerate(zip(data_sets, labels)):
        
        ovlp = data[:,0]
        var = data[:,1]
        ax.scatter(p,ovlp, color = colors[i], s = 75, marker = markers[i], 
                   edgecolor='black', lw=0.9, label=label)
        ax.plot(p, ovlp, '--', color = colors[i], lw = 0.75)
        ax.fill_between(p,ovlp-var,ovlp+var, color = colors[i], alpha=0.2)
        
    ax.set_xticks(p)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_xlabel('$p$', fontsize=22)
    ax.set_ylabel('$g(\psi)$', fontsize=22)
    ax.set_ylim([0.0,1.05])
    ax.legend(loc=loc,fontsize=16,framealpha=1.0,ncol=ncol,handleheight=1.0, 
              labelspacing=0.01,columnspacing=0.025)
    ax.grid(visible=True, which='major', axis='both',color='tab:gray', linestyle=(0, (1, 3)), linewidth=1.0, alpha=0.5)
    set_label_font(ax,16)
    plt.show()
    