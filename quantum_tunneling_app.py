import numpy as np
import scipy.constants as sc
from bokeh.plotting import figure 
from bokeh.layouts import row, column, widgetbox
from bokeh.models.widgets import Slider, Div, Button, RadioButtonGroup
from bokeh.events import ButtonClick
from bokeh.server.server import Server
from bokeh.models import Label
from bokeh.palettes import Colorblind

# Class for Quantum Tunneling
class Qtunnel:
    
    # Initializations
    def __init__(self, V0, bw, ke, sig):
        self.V0 = V0 * sc.value('electron volt')  # height of potential barrier in Joules
        self.ke = ke * sc.value('electron volt')  # kinetic energy of electron in Joules
        self.k0 = np.sqrt(self.ke * 2 * sc.m_e / (sc.hbar**2))  # wave vector of electron in m^-1
        self.bw = bw * sc.value('Angstrom star')  # potential barrier width in m
        self.sig = sig * sc.value('Angstrom star')  # Initial spread of Gaussian wavefunction
        self.dx = np.minimum((self.bw / 25.0), (self.sig / 25.0))  # grid cell size
        self.dt = 0.9 * sc.hbar / ((sc.hbar**2/(sc.m_e * self.dx**2)) + (self.V0 / 2.0))  # time step size
        length = 40 * np.maximum(self.bw, self.sig)  # length of the simulation domain
        self.ll = int(length / self.dx)  # total number of grid points in the domain
        vel = sc.hbar * self.k0 / sc.m_e
        self.tt = int(0.35 * length / vel / self.dt)  # total number of time steps in the simulation
        self.lx = np.linspace(0.0, length, self.ll)  # 1D position vector along x
        # potential barrier
        self.Vx = np.zeros(self.ll)
        bwgrid = int(self.bw/(2.0 * self.dx))
        bposgrid = int(self.ll/2.0)
        bl = bposgrid - bwgrid
        br = bposgrid + bwgrid
        self.Vx[bl:br] = self.V0
        # wavefunction arrays
        self.psir = np.zeros((self.ll))
        self.psii = np.zeros((self.ll))
        self.psimag = np.zeros(self.ll)
        ac = 1.0 / np.sqrt((np.sqrt(np.pi)) * self.sig)
        x0 = bl * self.dx - 6 * self.sig
        psigauss = ac * np.exp(-(self.lx - x0)**2 / (2.0 * self.sig**2))
        self.psir = psigauss * np.cos(self.k0 * self.lx)
        self.psii = psigauss * np.sin(self.k0 * self.lx)
        self.psimag = self.psir**2 + self.psii**2
        # fdtd update coefficients
        self.c1 = sc.hbar * self.dt / (2.0 * sc.m_e * self.dx**2)
        self.c2 = self.dt / sc.hbar
    
    # FDTD update for solving Schrodinger's equation
    def fdtd_update(self):
        self.psii[1:self.ll - 1] = (self.c1 * (self.psir[2:self.ll] - 2.0 * self.psir[1:self.ll - 1]
                                    + self.psir[0:self.ll - 2]) 
                                    - self.c2 * self.Vx[1:self.ll - 1] * self.psir[1:self.ll - 1]
                                    + self.psii[1:self.ll - 1])
        self.psir[1:self.ll - 1] = (-self.c1 * (self.psii[2:self.ll] - 2.0 * self.psii[1:self.ll - 1]
                                    + self.psii[0:self.ll - 2]) 
                                    + self.c2 * self.Vx[1:self.ll - 1] * self.psii[1:self.ll - 1]
                                    + self.psir[1:self.ll - 1])
        self.psimag = self.psir**2 + self.psii**2

    # Update plots
    def update_plots(self, r12):
        r12.data_source.data['y'] = self.psimag / np.amax(self.psimag)    


# Arrays for plotting


# Function to modify webpage (doc)
def modify_doc(doc):

    p1 = figure(plot_width=600, plot_height=500, title='Quantum Tunneling Animation')
    p1.xaxis.axis_label = 'position (Angstrom)'
    p1.yaxis.axis_label = 'Amplitude Squared (normalized)'
    p1.toolbar.logo = None
    p1.toolbar_location = None
    r11 = p1.line([], [], legend='Barrier', color=Colorblind[8][5], line_width=2)
    r12 = p1.line([], [], legend='Wavefunction', color=Colorblind[8][7], line_width=2)

    p2 = figure(plot_width=400, plot_height=250, title='Normalized wavefunctions at start')
    p2.xaxis.axis_label = 'position (Angstrom)'
    p2.yaxis.axis_label = 'Amplitude'
    p2.toolbar.logo = None
    p2.toolbar_location = None
    r21 = p2.line([], [], legend='Barrier', color=Colorblind[8][5])
    r22 = p2.line([], [], legend='Magnitude', color=Colorblind[8][7])
    r23 = p2.line([], [], legend='Real part', color=Colorblind[8][0])
    r24 = p2.line([], [], legend='Imag. part', color=Colorblind[8][6])
 
    p3 = figure(plot_width=400, plot_height=250, title='Normalized wavefunctions at end')
    p3.xaxis.axis_label = 'position (Angstrom)'
    p3.yaxis.axis_label = 'Amplitude'
    p3.toolbar.logo = None
    p3.toolbar_location = None
    r31 = p3.line([], [], color=Colorblind[8][5])
    r32 = p3.line([], [], color=Colorblind[8][7])
    r33 = p3.line([], [], color=Colorblind[8][0])
    r34 = p3.line([], [], color=Colorblind[8][6])

    barrier_height = Slider(title='Barrier Height (eV)', value=600, start=20, end=1000, step=100)
    barrier_width = Slider(title='Barrier Width (Angstrom)', value=0.3, start=0.3, end=1.0, step=0.1)
    electron_energy = Slider(title='Electron Energy (eV)', value=500, start=10, end=900, step=100)
    psi_spread = Slider(title='Wavefunction Spread (Angstrom)', value=0.8, start=0.3, end=1.0, step=0.1)
    startbutton = Button(label='Start', button_type='success')
    textdisp = Div(text='''<b>Note:</b> Wait for simulation  to stop before pressing buttons.''')

    def run_qt_sim(event):
    
        # Reset plots
        r21.data_source.data['x'] = []
        r22.data_source.data['x'] = []
        r23.data_source.data['x'] = []
        r24.data_source.data['x'] = []
        r21.data_source.data['y'] = []      
        r22.data_source.data['y'] = []
        r23.data_source.data['y'] = []
        r24.data_source.data['y'] = []
        r31.data_source.data['x'] = []
        r32.data_source.data['x'] = []
        r33.data_source.data['x'] = []
        r34.data_source.data['x'] = []
        r31.data_source.data['y'] = []
        r32.data_source.data['y'] = []
        r33.data_source.data['y'] = []
        r34.data_source.data['y'] = []

        # Get widget values
        V0 = barrier_height.value
        bw = barrier_width.value
        ke = electron_energy.value
        sig = psi_spread.value

        # Create Qtunnel object
        qt = Qtunnel(V0, bw, ke, sig)
        
        # Plot initial states
        r21.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r22.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r23.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r24.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r21.data_source.data['y'] = qt.Vx / np.amax(qt.Vx)
        r22.data_source.data['y'] = qt.psimag / np.amax(qt.psimag)
        r23.data_source.data['y'] = qt.psir / np.amax(qt.psir)
        r24.data_source.data['y'] = qt.psii / np.amax(qt.psii) 
        r11.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r12.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r11.data_source.data['y'] = qt.Vx / np.amax(qt.Vx)

        for n in range(qt.tt):
            qt.fdtd_update()
            qt.update_plots(r12)


        # Plot final states
        r31.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r32.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r33.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r34.data_source.data['x'] = qt.lx / sc.value('Angstrom star')
        r31.data_source.data['y'] = qt.Vx / np.amax(qt.Vx)
        r32.data_source.data['y'] = qt.psimag / np.amax(qt.psimag)
        r33.data_source.data['y'] = qt.psir / np.amax(qt.psir)
        r34.data_source.data['y'] = qt.psii / np.amax(qt.psii) 

    # Setup callbacks
    startbutton.on_event(ButtonClick, run_qt_sim)
    doc.add_root(column(row(barrier_height, barrier_width, textdisp), row(electron_energy, psi_spread, startbutton),
                 row(p1, column(p2, p3)))) 

server = Server({'/': modify_doc}, num_procs=1)
server.start()
 
if __name__=='__main__':
    print('Opening Bokeh application on http://localhost:5006/')
    server.io_loop.add_callback(server.show, '/')
    server.io_loop.start()
